#!/usr/bin/env python3
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from tqdm import tqdm

from wikisql_trimqa_common import load_jsonl_by_id, parse_args, read_jsonl, row_text, write_jsonl


def item_id(obj):
    return str(obj.get("id") or obj.get("item_id"))


def load_same_header_fillers(tables_path: str = None) -> Dict[str, List[List[Any]]]:
    fillers: Dict[str, List[List[Any]]] = defaultdict(list)
    if not tables_path:
        return fillers
    for table in read_jsonl(tables_path):
        key = "\t".join(str(x) for x in table.get("header", []))
        for row in table.get("rows", []):
            fillers[key].append(row)
    return fillers


def make_local_filler(item: Dict[str, Any], n: int) -> List[List[str]]:
    return [[f"filler_{n}_{j}" for j, _ in enumerate(item["header"])] for _ in range(n)]


def choose_fillers(item: Dict[str, Any], n: int, source: str, same_header: Dict[str, List[List[Any]]], seed: int) -> Tuple[List[List[Any]], int]:
    if source == "blank":
        return [["" for _ in item["header"]] for _ in range(n)], 0
    if source == "local_synthetic":
        return make_local_filler(item, n), 0
    key = "\t".join(str(x) for x in item["header"])
    pool = [row for row in same_header.get(key, []) if row not in item["rows"]]
    rng = random.Random(f"{seed}:{item_id(item)}:fillers")
    rng.shuffle(pool)
    if len(pool) >= n:
        return pool[:n], n
    return [*pool, *make_local_filler(item, n - len(pool))], len(pool)


def matched_ctrl_rows(
    item: Dict[str, Any],
    scores: Sequence[float],
    treat_rows: Sequence[int],
    trimqa_kept: Sequence[int],
    delta: float,
    len_tol: float,
) -> Tuple[List[int], bool]:
    treat_set = set(treat_rows)
    kept_low = [idx for idx in trimqa_kept if idx in set(item["low_match_rows"]) and idx not in treat_set]
    selected = []
    used = set()
    for t in treat_rows:
        t_score = scores[t]
        t_len = len(item["row_texts"][t])
        length_tol = len_tol * max(1, t_len) if len_tol < 1 else len_tol
        candidates = [
            idx for idx in kept_low
            if idx not in used
            and abs(scores[idx] - t_score) <= delta
            and abs(len(item["row_texts"][idx]) - t_len) <= length_tol
        ]
        if not candidates:
            candidates = [idx for idx in kept_low if idx not in used]
        if not candidates:
            return selected, False
        best = min(candidates, key=lambda idx: (abs(scores[idx] - t_score), abs(len(item["row_texts"][idx]) - t_len), idx))
        selected.append(best)
        used.add(best)
    return selected, True


def replace_rows(item: Dict[str, Any], selected_rows: Sequence[int], fillers: Sequence[Sequence[Any]]) -> Dict[str, Any]:
    rows = [list(row) for row in item["rows"]]
    for idx, filler in zip(selected_rows, fillers):
        rows[idx] = list(filler)
    out = dict(item)
    out["rows"] = rows
    out["row_texts"] = [row_text(item["header"], row) for row in rows]
    return out


def main():
    p = parse_args()
    p.add_argument("--items", required=True)
    p.add_argument("--selections", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--selector", default="trimqa_v1")
    p.add_argument("--filler_source", choices=["same_header_tables", "blank", "local_synthetic"], default="same_header_tables")
    p.add_argument("--tables")
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--len_tol", type=float, default=40)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()

    items = load_jsonl_by_id(args.items)
    selections = load_jsonl_by_id(args.selections)
    same_header = load_same_header_fillers(args.tables)
    variants = []
    reasons = Counter()

    for item_key, item in tqdm(items.items(), desc="making variants"):
        sel = selections[item_key]
        selected_kept = sel["selectors"].get(args.selector)
        if selected_kept is None:
            raise KeyError(f"selector {args.selector!r} missing for {item_key}")
        kept_set = set(selected_kept)
        if item["gold_row"] not in kept_set:
            reasons["gold_not_kept"] += 1
            continue
        treat_rows = [idx for idx in item["competing_near_match_rows"] if idx in kept_set]
        if not treat_rows:
            reasons["no_kept_competitor"] += 1
            continue
        ctrl_rows, matched = matched_ctrl_rows(item, sel["scores"], treat_rows, selected_kept, args.delta, args.len_tol)
        if not ctrl_rows:
            reasons["no_ctrl_rows"] += 1
            continue
        fillers, n_foreign = choose_fillers(item, len(treat_rows), args.filler_source, same_header, args.seed)
        base = dict(item)
        base["variant"] = "base"
        treat = replace_rows(item, treat_rows, fillers)
        treat["variant"] = "treat"
        ctrl = replace_rows(item, ctrl_rows, fillers)
        ctrl["variant"] = "ctrl"
        reasons["built"] += 1
        variants.append({
            "id": item_key,
            "item_id": item_key,
            "selector": args.selector,
            "selected_kept_rows": selected_kept,
            "matched": matched,
            "delta": args.delta,
            "len_tol": args.len_tol,
            "filler_source": args.filler_source,
            "with_foreign_fillers": n_foreign > 0,
            "n_foreign_fillers": n_foreign,
            "treat_rows": treat_rows,
            "ctrl_rows": ctrl_rows,
            "treat_scores": [sel["scores"][idx] for idx in treat_rows],
            "ctrl_scores": [sel["scores"][idx] for idx in ctrl_rows],
            "treat_lengths": [len(item["row_texts"][idx]) for idx in treat_rows],
            "ctrl_lengths": [len(item["row_texts"][idx]) for idx in ctrl_rows],
            "variants": {"base": base, "treat": treat, "ctrl": ctrl},
        })

    write_jsonl(args.out, variants)
    summary_path = Path(args.out).with_name(Path(args.out).stem + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        import json
        json.dump({
            "selector": args.selector,
            "items": len(items),
            "built": len(variants),
            "reasons": dict(sorted(reasons.items())),
        }, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

