#!/usr/bin/env python3
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from wikisql_trimqa_common import (
    canonical_answer,
    execute_noagg,
    load_wikisql_tables,
    matching_condition_count,
    parse_args,
    read_jsonl,
    row_text,
    values_equal,
    write_jsonl,
)


NUM_RE = re.compile(r"[-+]?\d*\.\d+|\d+")
OPS = {0: "=", 1: ">", 2: "<"}


def sqlite_table_name(table_id):
    return "table_" + str(table_id).replace("-", "_")


def parse_real(value):
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", "").strip()
    match = NUM_RE.search(text)
    if not match:
        raise ValueError(f"no numeric value in {value!r}")
    return float(match.group(0))


def sqlite_column_types(conn, table_id):
    table_name = sqlite_table_name(table_id)
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1]: row[2].lower() for row in rows}


def execute_sqlite_noagg(conn, table_id, sql):
    table_name = sqlite_table_name(table_id)
    types = sqlite_column_types(conn, table_id)
    where = []
    params = []
    for col, op, value in sql.get("conds") or []:
        col_name = f"col{int(col)}"
        col_type = types.get(col_name, "text")
        if isinstance(value, str):
            value = value.lower()
        if col_type == "real" and not isinstance(value, (int, float)):
            value = parse_real(value)
        where.append(f"{col_name} {OPS[int(op)]} ?")
        params.append(value)
    where_sql = " WHERE " + " AND ".join(where) if where else ""
    query = f"SELECT col{int(sql['sel'])} FROM {table_name}{where_sql}"
    return [row[0] for row in conn.execute(query, params).fetchall()]


def row_sets(example, table, conn=None):
    sql = example.get("sql") or {}
    conds = sql.get("conds") or []
    support = execute_noagg(table, sql)
    if len(support) != 1:
        return None
    support_idx, matcher_answer = support[0]
    sqlite_answers = execute_sqlite_noagg(conn, table.table_id, sql) if conn is not None else [matcher_answer]
    executor_answer = canonical_answer(sqlite_answers[0]) if len(sqlite_answers) == 1 else None
    sel = int(sql["sel"])
    k = len(conds)
    match_counts = [matching_condition_count(row, conds, table.types) for row in table.rows]
    competitors = []
    low_match = []
    support_answer = canonical_answer(table.rows[support_idx][sel])
    for idx, row in enumerate(table.rows):
        if idx == support_idx:
            continue
        answer = canonical_answer(row[sel]) if 0 <= sel < len(row) else ""
        if match_counts[idx] == k - 1 and answer and answer != support_answer:
            competitors.append(idx)
        elif match_counts[idx] <= max(0, k - 2):
            low_match.append(idx)
    executor_agrees = (
        len(sqlite_answers) == 1
        and values_equal(sqlite_answers[0], table.rows[support_idx][sel], table.types[sel] if table.types else "text")
    )
    return support_idx, executor_answer, support_answer, match_counts, competitors, low_match, executor_agrees


def build_item(example, table, support_idx, support_answer, match_counts, competitors, low_match):
    item_id = f"{example.get('phase', 'wikisql')}:{example.get('table_id')}:{example.get('question')}"
    return {
        "id": item_id,
        "item_id": item_id,
        "question": example["question"],
        "table_id": table.table_id,
        "table_name": table.name,
        "header": table.header,
        "rows": table.rows,
        "n_rows": len(table.rows),
        "n_conds": len(example["sql"].get("conds") or []),
        "sql": example["sql"],
        "answer": support_answer,
        "support_row": support_idx,
        "gold_row": support_idx,
        "competing_near_match_rows": competitors,
        "low_match_rows": low_match,
        "row_match_counts": match_counts,
        "row_texts": [row_text(table.header, row) for row in table.rows],
    }


def main():
    p = parse_args()
    p.add_argument("--wikisql", required=True)
    p.add_argument("--split", required=True, choices=["train", "dev", "test"])
    p.add_argument("--min_rows", type=int, default=16)
    p.add_argument("--out", required=True)
    p.add_argument("--stats_out")
    args = p.parse_args()

    tables = load_wikisql_tables(args.wikisql, args.split)
    examples_path = Path(args.wikisql) / f"{args.split}.jsonl"
    db_path = Path(args.wikisql) / f"{args.split}.db"
    conn = sqlite3.connect(str(db_path)) if db_path.exists() else None
    stats = Counter()
    items = []

    for ex in tqdm(read_jsonl(examples_path), desc="building items"):
        stats["all"] += 1
        table = tables.get(str(ex.get("table_id")))
        if table is None:
            continue
        sql = ex.get("sql") or {}
        if int(sql.get("agg", -1)) != 0:
            continue
        stats["no_agg"] += 1
        if len(sql.get("conds") or []) < 2:
            continue
        stats["ge2_conds"] += 1

        sets = row_sets(ex, table, conn)
        if sets is None:
            continue
        stats["unique_support"] += 1
        support_idx, executor_answer, support_answer, match_counts, competitors, low_match, executor_agrees = sets
        supplied_answer = ex.get("answer")
        supplied_disagrees = supplied_answer is not None and canonical_answer(supplied_answer) != support_answer
        executor_disagrees = (not executor_agrees) or supplied_disagrees
        if executor_agrees and not supplied_disagrees:
            stats["executor_verified"] += 1
        if not competitors:
            continue
        stats["has_competing_near_match"] += 1
        if len(table.rows) < args.min_rows:
            continue
        stats[f"n_rows>={args.min_rows}"] += 1
        if not low_match:
            stats["drop_no_low_match_rows"] += 1
            continue
        if executor_disagrees:
            stats["drop_executor_disagrees"] += 1
            continue
        items.append(build_item(ex, table, support_idx, support_answer, match_counts, competitors, low_match))

    stats["kept"] = len(items)
    if items:
        n_rows = sorted(len(x["rows"]) for x in items)
        near = sorted(len(x["competing_near_match_rows"]) for x in items)
        stats["n_rows_median"] = n_rows[len(n_rows) // 2]
        stats["n_rows>=30"] = sum(1 for x in items if len(x["rows"]) >= 30)
        stats["near_match_rows_median"] = near[len(near) // 2]
        stats["tables_unique"] = len({x["table_id"] for x in items})

    write_jsonl(args.out, items)
    stats_path = args.stats_out or str(Path(args.out).with_suffix(".stats.json"))
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(dict(stats), fh, indent=2, sort_keys=True)
    print(json.dumps(dict(stats), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

