#!/usr/bin/env python3
import json
from answer_metric import answer_match, exact_match
from collections import Counter, defaultdict
from typing import Any, Dict, List

from wikisql_trimqa_common import (
    bootstrap_mean_diff,
    canonical_answer,
    load_jsonl_by_id,
    mcnemar_exact,
    mean,
    n_needed_for_power,
    parse_args,
    read_jsonl,
)


def row_id(row):
    return str(row.get("id") or row.get("item_id"))


def row_source(row):
    return str(row.get("source") or row.get("condition"))


def rate_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "correct": None,
            "strict_em": None,
            "competitor_answer": None,
            "competitor_attributable": None,
        }
    return {
        "n": len(rows),
        "correct": sum(bool(r["correct"]) for r in rows) / len(rows),
        "strict_em": mean([float(r.get("strict_em", False)) for r in rows]),
        "competitor_answer": mean([float(r.get("competitor_answer", False)) for r in rows]),
        "competitor_attributable": mean([
            float(r.get("competitor_answer", False) and not r.get("correct", False))
            for r in rows
        ]),
    }


def attach_competitor_flags(answer_rows, items):
    for row in answer_rows:
        item = items[row_id(row)]
        competitor_answers = {
            canonical_answer(item["rows"][idx][item["sql"]["sel"]])
            for idx in item["competing_near_match_rows"]
        }
        row["competitor_answer"] = canonical_answer(row["answer"]) in competitor_answers


def selection_audit(selections):
    totals = defaultdict(Counter)
    for sel in selections.values():
        for name, audit in sel.get("audit", {}).items():
            totals[name]["competitor_kept"] += audit["competitor_kept_count"]
            totals[name]["competitor_total"] += audit["competitor_total"]
            totals[name]["low_kept"] += audit["low_kept_count"]
            totals[name]["low_total"] += audit["low_total"]
            totals[name]["support_kept"] += int(audit["support_kept"])
            totals[name]["items"] += 1
    out = {}
    for name, c in totals.items():
        out[name] = {
            "P(kept|competing_near_match_row)": c["competitor_kept"] / c["competitor_total"] if c["competitor_total"] else None,
            "P(kept|low_match_row)": c["low_kept"] / c["low_total"] if c["low_total"] else None,
            "P(support_kept)": c["support_kept"] / c["items"] if c["items"] else None,
        }
    return out


def paired_contrast(ids, left, right, left_name, right_name, n_boot, seed):
    ids = sorted(ids)
    diffs = [(i, float(left[i]["correct"]) - float(right[i]["correct"])) for i in ids]
    right_only = sum((not left[i]["correct"]) and right[i]["correct"] for i in ids)
    left_only = sum(left[i]["correct"] and (not right[i]["correct"]) for i in ids)
    boot = bootstrap_mean_diff(diffs, n_boot, seed)
    effect = boot["mean"] or 0.0
    discordant_rate = (right_only + left_only) / len(ids) if ids else 0.0
    return {
        **boot,
        f"discordant_{left_name}_wrong_{right_name}_right": right_only,
        f"discordant_{left_name}_right_{right_name}_wrong": left_only,
        "mcnemar_p": mcnemar_exact(right_only, left_only),
        "n_needed_for_80pct_power": n_needed_for_power(discordant_rate, effect),
    }


def exp2_contrast(ids, treat, ctrl, n_boot, seed):
    return paired_contrast(ids, treat, ctrl, "treat", "ctrl", n_boot, seed)


def variant_mean_score(v, key):
    vals = v.get(key) or []
    return mean(vals) if vals else None


def exp2_by_model(answer_rows, variants, n_boot, seed):
    by_model_source = defaultdict(list)
    for row in answer_rows:
        by_model_source[(row["model"], row_source(row))].append(row)

    variant_by_id = {row_id(v): v for v in variants}
    treat_le_ctrl = {
        i for i, v in variant_by_id.items()
        if v.get("matched") and variant_mean_score(v, "treat_scores") is not None
        and variant_mean_score(v, "ctrl_scores") is not None
        and variant_mean_score(v, "treat_scores") <= variant_mean_score(v, "ctrl_scores")
    }
    treat_gt_ctrl = {
        i for i, v in variant_by_id.items()
        if v.get("matched") and variant_mean_score(v, "treat_scores") is not None
        and variant_mean_score(v, "ctrl_scores") is not None
        and variant_mean_score(v, "treat_scores") > variant_mean_score(v, "ctrl_scores")
    }

    out = {}
    models = sorted({row["model"] for row in answer_rows})
    for model in models:
        base = {row_id(r): r for r in by_model_source[(model, "var:base")]}
        treat = {row_id(r): r for r in by_model_source[(model, "var:treat")]}
        ctrl = {row_id(r): r for r in by_model_source[(model, "var:ctrl")]}
        matched = set(base) & set(treat) & set(ctrl)
        gold_only_rows = (
            by_model_source[(model, "sel:external:gold_only")]
            or by_model_source[(model, "sel:gold_only")]
        )
        gold_only = {row_id(r): r for r in gold_only_rows}
        answerable = {i for i, r in gold_only.items() if r.get("correct")}
        out[model] = {
            "similarity_matched": {
                "PRIMARY treat_vs_ctrl": exp2_contrast(matched, treat, ctrl, n_boot, seed),
                "matched&gold_only_correct treat_vs_ctrl": exp2_contrast(matched & answerable, treat, ctrl, n_boot, seed),
                "treat_vs_base": paired_contrast(matched, treat, base, "treat", "base", n_boot, seed),
                "ctrl_vs_base": paired_contrast(matched, ctrl, base, "ctrl", "base", n_boot, seed),
                "treat_scores<=ctrl_scores treat_vs_ctrl": exp2_contrast(matched & treat_le_ctrl, treat, ctrl, n_boot, seed),
                "treat_scores>ctrl_scores treat_vs_ctrl": exp2_contrast(matched & treat_gt_ctrl, treat, ctrl, n_boot, seed),
            }
        }
    return out


def exp2_balance(variants):
    return {
        "matched_rate": mean([float(v["matched"]) for v in variants]),
        "treat_score_mean": mean([x for v in variants for x in v["treat_scores"]]),
        "ctrl_score_mean": mean([x for v in variants for x in v["ctrl_scores"]]),
        "treat_length_mean": mean([x for v in variants for x in v["treat_lengths"]]),
        "ctrl_length_mean": mean([x for v in variants for x in v["ctrl_lengths"]]),
        "with_foreign_fillers": any(v.get("with_foreign_fillers") for v in variants),
        "n_foreign_fillers": sum(int(v.get("n_foreign_fillers", 0)) for v in variants),
    }


def build_parser():
    p = parse_args()
    p.add_argument("--items", required=True)
    p.add_argument("--answers", required=True)
    p.add_argument("--selections")
    p.add_argument("--variants")
    p.add_argument("--out", required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=23)
    return p


def main(args=None):
    if args is None:
        args = build_parser().parse_args()
    if not hasattr(args, "bootstrap"):
        args.bootstrap = 2000
    if not hasattr(args, "seed"):
        args.seed = 23

    items = load_jsonl_by_id(args.items)
    selections = load_jsonl_by_id(args.selections) if getattr(args, "selections", None) else {}
    answer_rows = list(read_jsonl(args.answers))
    for row in answer_rows:
        item = items[row_id(row)]
        pred = row.get("prediction", row.get("answer", ""))
        gold = row.get("gold", item.get("answer", ""))
        row["strict_em"] = exact_match(pred, gold)
        row["correct"] = answer_match(pred, gold)
    attach_competitor_flags(answer_rows, items)
    variants = list(read_jsonl(args.variants)) if getattr(args, "variants", None) else []

    grouped = defaultdict(list)
    for row in answer_rows:
        grouped[(row["model"], row_source(row))].append(row)

    report = {
        "rates": {
            model: {source: rate_summary(rows) for (m, source), rows in grouped.items() if m == model}
            for model in sorted({m for m, _ in grouped})
        },
        "selection_audit": selection_audit(selections) if selections else {},
        "exp2": exp2_by_model(answer_rows, variants, args.bootstrap, args.seed),
        "exp2_balance": exp2_balance(variants) if variants else {},
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
