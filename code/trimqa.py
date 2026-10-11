#!/usr/bin/env python3
import importlib.util
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

from tqdm import tqdm

from wikisql_trimqa_common import make_encoder, parse_args, read_jsonl, row_text, write_jsonl


def item_id(obj: Dict[str, Any]) -> str:
    return str(obj.get("id") or obj.get("item_id"))


def load_external_selectors(specs: List[str]) -> Dict[str, Dict[str, List[int]]]:
    externals: Dict[str, Dict[str, List[int]]] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"--external must be name=path, got: {spec}")
        name, path = spec.split("=", 1)
        rows = {}
        for obj in read_jsonl(path):
            key = str(obj.get("id") or obj.get("item_id"))
            kept = obj.get("kept_rows") or obj.get("row_indices") or obj.get("kept_row_ids") or []
            rows[key] = sorted(int(x) for x in kept)
        externals[name] = rows
    return externals


class TrimQAv1:
    def __init__(self, trimqa_repo: str = None):
        self.config = {
            "keep_ratio": 0.40,
            "superlative_keep_ratio": 0.50,
            "temporal_keep_ratio": 0.50,
            "episode_keep_ratio": 0.45,
            "cast_lookup_keep_ratio": 0.45,
            "list_enum_keep_ratio": 0.50,
            "comprehensive_keep_ratio": 0.60,
            "boundary_count": 5,
            "min_rows_to_keep": 3,
            "small_table_threshold": 5,
        }
        self.row_to_text = None
        if trimqa_repo:
            self._load_repo_helpers(trimqa_repo)

    def _load_repo_helpers(self, trimqa_repo: str) -> None:
        candidates = [
            Path(trimqa_repo) / "scripts" / "Pruning" / "sbert_pruner.py",
            Path(trimqa_repo) / "scripts" / "Pruning" / "colbert_pruner_NQ.py",
        ]
        for path in candidates:
            if not path.exists():
                continue
            spec = importlib.util.spec_from_file_location("_trimqa_v1_source", path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                configs = getattr(module, "VERSION_CONFIGS", {})
                if "v1" in configs:
                    self.config.update(configs["v1"])
                self.row_to_text = getattr(module, "row_to_text", None)
                return

    def row_chunks(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        columns = " | ".join(item["header"])
        return [
            {
                "chunk_id": f"{item['table_id']}_row_{idx}",
                "table_id": item["table_id"],
                "document_title": item.get("table_name", ""),
                "columns": columns,
                "row": " | ".join("" if cell is None else str(cell) for cell in row),
                "row_index": idx,
            }
            for idx, row in enumerate(item["rows"])
        ]

    def query_type_and_ratio(self, question: str) -> Tuple[str, float]:
        q = question.lower()
        words = set(re.findall(r"\w+", q))
        cfg = self.config
        if any(x in q for x in cfg.get("comprehensive_patterns", set())) or any(x in q for x in ("list all", "name all", "how many", "all of the")):
            return "comprehensive", cfg["comprehensive_keep_ratio"]
        if any(x in q for x in cfg.get("episode_air_patterns", set())):
            return "episode_air", cfg.get("episode_keep_ratio", 0.60)
        if words.intersection({"won", "winner", "sang", "performed", "current", "year"}) or any(x in q for x in ("last time", "end of", "beginning of")):
            return "temporal", cfg["temporal_keep_ratio"]
        if words.intersection({"highest", "lowest", "largest", "smallest", "biggest", "shortest", "tallest", "most", "least", "best", "worst", "greatest", "maximum", "minimum", "top", "bottom", "first", "last", "recent", "latest", "oldest", "newest", "earliest", "longest", "fastest", "slowest", "only", "fewest"}):
            return "superlative", cfg["superlative_keep_ratio"]
        if any(x in q for x in ("which are", "what are", "who are", "list", "name the")):
            return "list_enum", cfg.get("list_enum_keep_ratio", cfg["keep_ratio"])
        if any(x in q for x in ("episode", "series", "season")):
            return "episode", cfg["episode_keep_ratio"]
        if any(x in q for x in ("real name", "actor", "actress", "played", "plays", "portrayed")):
            return "cast_lookup", cfg.get("cast_lookup_keep_ratio", cfg["keep_ratio"])
        return "base", cfg["keep_ratio"]

    def trimqa_select(self, item: Dict[str, Any], scores: Sequence[float]) -> List[int]:
        n_rows = len(item["rows"])
        if n_rows <= self.config["small_table_threshold"]:
            return list(range(n_rows))
        query_type, keep_ratio = self.query_type_and_ratio(item["question"])
        keep_count = min(max(self.config["min_rows_to_keep"], int(n_rows * keep_ratio)), n_rows)
        forced = set(range(min(3, n_rows)))
        if query_type != "base":
            bc = min(self.config["boundary_count"], n_rows // 2)
            forced.update(range(bc))
            forced.update(range(n_rows - bc, n_rows))
        extremes = detect_numeric_extremes(item["rows"])
        forced.update(extremes["max_rows"])
        forced.update(extremes["min_rows"])
        return sorted(set(topk(scores, keep_count)).union(forced))


def detect_numeric_extremes(rows: List[List[Any]]) -> Dict[str, Set[int]]:
    row_numbers = []
    for idx, row in enumerate(rows):
        numbers = []
        for cell in row:
            for num_str in re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b", str(cell)):
                try:
                    numbers.append(float(num_str.replace(",", "").replace("%", "")))
                except ValueError:
                    pass
        row_numbers.append((idx, numbers))
    all_numbers = [n for _, nums in row_numbers for n in nums]
    if len(all_numbers) < 3:
        return {"max_rows": set(), "min_rows": set()}
    sorted_nums = sorted(all_numbers)
    top_threshold = sorted_nums[-3]
    bottom_threshold = sorted_nums[2]
    max_rows, min_rows = set(), set()
    for idx, nums in row_numbers:
        if any(n >= top_threshold for n in nums):
            max_rows.add(idx)
        if any(n <= bottom_threshold for n in nums):
            min_rows.add(idx)
    return {"max_rows": max_rows, "min_rows": min_rows}


def topk(scores: Sequence[float], k: int) -> List[int]:
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


def evidence_preserving_random(item: Dict[str, Any], budget: int, seed: int) -> List[int]:
    import random

    rng = random.Random(f"{seed}:{item_id(item)}")
    required: Set[int] = {item["support_row"]}
    rows = list(range(len(item["rows"])))
    if len(required) >= budget:
        return sorted(required)
    pool = [idx for idx in rows if idx not in required]
    rng.shuffle(pool)
    return sorted([*required, *pool[: budget - len(required)]])


def audit_for_selector(item: Dict[str, Any], kept: Sequence[int]) -> Dict[str, Any]:
    kept_set = set(kept)
    comp = item["competing_near_match_rows"]
    low = item["low_match_rows"]
    return {
        "n_kept": len(kept),
        "support_kept": item["support_row"] in kept_set,
        "competitor_kept_count": sum(1 for idx in comp if idx in kept_set),
        "competitor_total": len(comp),
        "low_kept_count": sum(1 for idx in low if idx in kept_set),
        "low_total": len(low),
    }


def main():
    p = parse_args()
    p.add_argument("--items", required=True)
    p.add_argument("--trimqa_repo")
    p.add_argument("--out", required=True)
    p.add_argument("--encoder", choices=["sentence-transformers", "hash"], default="sentence-transformers")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--tap4llm")
    p.add_argument("--external", action="append", default=[], help="Extra selector as name=path JSONL with item_id/id and kept_rows")
    args = p.parse_args()

    items = list(read_jsonl(args.items))
    trimqa = TrimQAv1(args.trimqa_repo)
    encoder = make_encoder(args.encoder, args.model)
    tap = {str(obj.get("id") or obj.get("item_id")): obj for obj in read_jsonl(args.tap4llm)} if args.tap4llm else {}
    externals = load_external_selectors(args.external)

    records = []
    for item in tqdm(items, desc="selecting rows"):
        docs = [row_text(item["header"], row) for row in item["rows"]]
        scores = encoder.score_rows(item["question"], docs)
        trimqa_kept = trimqa.trimqa_select(item, scores)
        oracle = evidence_preserving_random(item, len(trimqa_kept), args.seed)
        selectors = {
            "full": list(range(len(item["rows"]))),
            "trimqa_v1": trimqa_kept,
            "sim_topk": sorted(topk(scores, len(trimqa_kept))),
            "oracle_random": oracle,
            "random_oracle": oracle,
        }
        key = item_id(item)
        if key in tap:
            selectors["tap4llm"] = sorted(tap[key].get("kept_rows") or tap[key].get("row_indices") or [])
        for name, rows_by_item in externals.items():
            if key in rows_by_item:
                selectors[name] = rows_by_item[key]

        records.append({
            "id": key,
            "item_id": key,
            "table_id": item["table_id"],
            "budget": len(trimqa_kept),
            "scores": scores,
            "selectors": selectors,
            "audit": {name: audit_for_selector(item, kept) for name, kept in selectors.items()},
        })

    write_jsonl(args.out, records)


if __name__ == "__main__":
    main()

