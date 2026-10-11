#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from answer_metric import answer_match, exact_match
from wikisql_trimqa_common import canonical_answer, mcnemar_exact, read_jsonl


def rid(row):
    return str(row.get("id") or row.get("item_id"))


def source(row):
    return str(row.get("source") or row.get("condition"))


def pct(x):
    return None if x is None else 100.0 * x


def load_jsonl(path):
    return [r for r in read_jsonl(path)]


def binom_two_sided(k, n, p=0.5):
    if n == 0:
        return None
    probs = [math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
    obs = probs[k]
    return min(1.0, sum(q for q in probs if q <= obs + 1e-15))


def score_answer(row, item):
    pred = row.get("prediction", row.get("answer", ""))
    gold = row.get("gold", item.get("answer", item.get("gold_answer", "")))
    row = dict(row)
    row["correct"] = answer_match(pred, gold)
    row["strict_em"] = exact_match(pred, gold)
    row["prediction_text"] = pred
    return row


def competitor_values(item):
    sel = item["sql"]["sel"]
    return {canonical_answer(item["rows"][idx][sel]) for idx in item.get("competing_near_match_rows", [])}


def any_value_collision(item, value, excluded_rows=()):
    cv = canonical_answer(value)
    excluded = set(excluded_rows)
    hits = []
    for i, row in enumerate(item["rows"]):
        if i in excluded:
            continue
        for j, cell in enumerate(row):
            if canonical_answer(cell) == cv:
                hits.append((i, j))
    return hits


def attach_flags(rows, items):
    out = []
    for row in rows:
        item = items[rid(row)]
        row = score_answer(row, item)
        row["competitor_answer"] = canonical_answer(row["prediction_text"]) in competitor_values(item)
        out.append(row)
    return out


def rate(rows, key="correct"):
    return sum(bool(r.get(key)) for r in rows) / len(rows) if rows else None


def paired(left, right, key="correct"):
    ids = sorted(set(left) & set(right))
    diffs = [float(left[i].get(key, False)) - float(right[i].get(key, False)) for i in ids]
    pos = sum(d > 0 for d in diffs)
    neg = sum(d < 0 for d in diffs)
    same_pos = sum(left[i].get(key, False) and right[i].get(key, False) for i in ids)
    same_neg = sum((not left[i].get(key, False)) and (not right[i].get(key, False)) for i in ids)
    return {
        "n": len(ids),
        "mean": sum(diffs) / len(diffs) if diffs else None,
        "left_right": pos,
        "right_left": neg,
        "same_right": same_pos,
        "same_wrong": same_neg,
        "mcnemar_p": mcnemar_exact(neg, pos),
        "sign_flip_p": binom_two_sided(pos, pos + neg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    items = {rid(r): r for r in load_jsonl(args.items)}
    variants = {rid(r): r for r in load_jsonl(args.variants)}
    answers = attach_flags(load_jsonl(args.answers), items)
    by_model_cond = defaultdict(dict)
    for row in answers:
        by_model_cond[(row["model"], source(row))][rid(row)] = row

    matched_ids = sorted(variants)
    expected = {
        "gpt-4o-2024-05-13": 0.025714285714285714,
        "Qwen/Qwen2.5-7B-Instruct": 0.05285714285714286,
        "meta-llama/Llama-3.1-8B-Instruct": 0.05142857142857143,
    }

    summary = {
        "inputs": {"items": args.items, "variants": args.variants, "answers": args.answers},
        "n_items": len(items),
        "n_built_or_matched_variants": len(variants),
        "tables": {
            "n_tables": len({it.get("table_id") for it in items.values()}),
            "questions_per_table": dict(Counter(it.get("table_id") for it in items.values())),
        },
        "models": {},
        "value_collisions": {},
    }

    collision_counts = Counter()
    comp_match_records = []
    for item_id, var in variants.items():
        item = items[item_id]
        treat_rows = var.get("treat_rows", [])
        ctrl_rows = var.get("ctrl_rows", [])
        sel = item["sql"]["sel"]
        comp_rows = item.get("competing_near_match_rows", [])
        comp_vals = [item["rows"][i][sel] for i in comp_rows]
        for val in comp_vals:
            in_noncomp_original = bool(any_value_collision(item, val, excluded_rows=comp_rows))
            in_treat_table = False
            trows = var["variants"]["treat"]["rows"]
            cv = canonical_answer(val)
            for i, row in enumerate(trows):
                for j, cell in enumerate(row):
                    if canonical_answer(cell) == cv:
                        in_treat_table = True
            collision_counts["competitor_values_total"] += 1
            collision_counts["competitor_value_elsewhere_original"] += int(in_noncomp_original)
            collision_counts["competitor_value_still_in_treat_table"] += int(in_treat_table)
        if any(comp_vals):
            collision_counts["questions_with_competitor_values"] += 1
            collision_counts["questions_any_competitor_value_elsewhere_original"] += int(any(any_value_collision(item, v, excluded_rows=comp_rows) for v in comp_vals))
            collision_counts["questions_any_competitor_value_still_in_treat_table"] += int(any(
                canonical_answer(cell) in {canonical_answer(v) for v in comp_vals}
                for row in var["variants"]["treat"]["rows"] for cell in row
            ))

    summary["value_collisions"] = dict(collision_counts)
    if collision_counts["questions_with_competitor_values"]:
        summary["share_questions_competitor_value_still_in_treat_table"] = (
            collision_counts["questions_any_competitor_value_still_in_treat_table"] /
            collision_counts["questions_with_competitor_values"]
        )
    if collision_counts["competitor_values_total"]:
        summary["share_competitor_values_still_in_treat_table"] = (
            collision_counts["competitor_value_still_in_treat_table"] /
            collision_counts["competitor_values_total"]
        )

    blind_rows = []
    key_rows = []
    for model in sorted({m for m, _ in by_model_cond}):
        base = by_model_cond[(model, "var:base")]
        treat = by_model_cond[(model, "var:treat")]
        ctrl = by_model_cond[(model, "var:ctrl")]
        ids = sorted(set(matched_ids) & set(base) & set(treat) & set(ctrl))
        p_tc = paired({i: treat[i] for i in ids}, {i: ctrl[i] for i in ids}, "correct")
        p_tb = paired({i: treat[i] for i in ids}, {i: base[i] for i in ids}, "correct")
        p_cb = paired({i: ctrl[i] for i in ids}, {i: base[i] for i in ids}, "correct")
        strict = paired({i: treat[i] for i in ids}, {i: ctrl[i] for i in ids}, "strict_em")
        model_summary = {
            "n": len(ids),
            "base_accuracy": rate([base[i] for i in ids]),
            "treat_accuracy": rate([treat[i] for i in ids]),
            "ctrl_accuracy": rate([ctrl[i] for i in ids]),
            "base_competitor_rate": rate([base[i] for i in ids], "competitor_answer"),
            "treat_competitor_rate": rate([treat[i] for i in ids], "competitor_answer"),
            "ctrl_competitor_rate": rate([ctrl[i] for i in ids], "competitor_answer"),
            "primary_treat_vs_ctrl": p_tc,
            "treat_vs_base": p_tb,
            "ctrl_vs_base": p_cb,
            "strict_em_treat_vs_ctrl": strict,
        }
        if model in expected:
            model_summary["gate_expected_primary_mean"] = expected[model]
            model_summary["gate_abs_error"] = abs((p_tc["mean"] or 0.0) - expected[model])
        summary["models"][model] = model_summary

        rng = random.Random(f"{args.seed}:{model}")
        for i in ids:
            if bool(treat[i]["correct"]) == bool(ctrl[i]["correct"]):
                continue
            pair = [("treat", treat[i]), ("ctrl", ctrl[i])]
            rng.shuffle(pair)
            blind_rows.append({
                "model": model,
                "item_id": i,
                "question": items[i]["question"],
                "gold": items[i].get("answer", items[i].get("gold_answer", "")),
                "answer_a": pair[0][1]["prediction_text"],
                "answer_b": pair[1][1]["prediction_text"],
                "a_acceptable(y/n)": "",
                "b_acceptable(y/n)": "",
                "notes": "",
            })
            key_rows.append({
                "model": model,
                "item_id": i,
                "a_condition": pair[0][0],
                "b_condition": pair[1][0],
                "treat_correct": treat[i]["correct"],
                "ctrl_correct": ctrl[i]["correct"],
                "treat_strict_em": treat[i]["strict_em"],
                "ctrl_strict_em": ctrl[i]["strict_em"],
            })

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(outdir / "discordant_blind.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "item_id", "question", "gold", "answer_a", "answer_b", "a_acceptable(y/n)", "b_acceptable(y/n)", "notes"])
        w.writeheader(); w.writerows(blind_rows)
    with open(outdir / "discordant_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "item_id", "a_condition", "b_condition", "treat_correct", "ctrl_correct", "treat_strict_em", "ctrl_strict_em"])
        w.writeheader(); w.writerows(key_rows)

    print(json.dumps({
        "out": str(outdir),
        "gate_means": {m: summary["models"][m]["primary_treat_vs_ctrl"]["mean"] for m in summary["models"]},
        "gate_abs_errors": {m: summary["models"][m].get("gate_abs_error") for m in summary["models"]},
        "discordant_blind_rows": len(blind_rows),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
