#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict
import json
import time

from tqdm import tqdm

from wikisql_trimqa_common import canonical_answer, load_jsonl_by_id, parse_args, read_jsonl, table_text, write_jsonl


PROMPT = """Answer the following question based on the table provided.
Give a short, direct answer (a few words). Do not explain.

Question: {question}
Table:
{table}
Answer:"""


def mock_answer(question: str, table: str, answer: str) -> str:
    return answer


def _retry_after_seconds(exc, default: int) -> int:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            value = response.headers.get("Retry-After")
            if value:
                return max(1, int(float(value)))
        except Exception:
            pass
    return default


def _retryable_openai_error(exc) -> bool:
    status = getattr(exc, "status_code", None)
    return status in {402, 408, 409, 429, 500, 502, 503, 504}


def openai_answer(client, model: str, question: str, table: str, max_tokens: int, retries: int = 6) -> str:
    prompt = PROMPT.format(question=question, table=table)
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            if attempt >= retries or not _retryable_openai_error(exc):
                raise
            time.sleep(_retry_after_seconds(exc, min(120, 5 * (2 ** attempt))))
    raise RuntimeError("unreachable retry state")


def parse_selector_filters(values):
    filters = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                filters.append(part)
    return filters


def selector_allowed(name, filters):
    if not filters:
        return None
    aliases = {name, f"sel:{name}"}
    if name == "gold_only":
        aliases.update({"external:gold_only", "sel:external:gold_only"})
    for f in filters:
        if f in aliases:
            return f"sel:{f}" if f.startswith("external:") else f if f.startswith("sel:") else f"sel:{name}"
    return False


def load_done(path):
    done = set()
    out = Path(path)
    if not out.exists():
        return done
    for row in read_jsonl(out):
        done.add((str(row.get("id") or row.get("item_id")), row.get("model"), row.get("source") or row.get("condition")))
    return done


def make_jobs(items: Dict[str, Dict[str, Any]], selections: Dict[str, Dict[str, Any]], variants_path: str, closed_book: bool, selector_filters=None):
    selector_filters = selector_filters or []
    if selections:
        for item_key, item in items.items():
            sel = selections[item_key]
            if closed_book and not selector_filters:
                yield {"id": item_key, "source": "closed_book", "item": item, "row_indices": []}
            for name, kept in sel["selectors"].items():
                label = selector_allowed(name, selector_filters)
                if label is False:
                    continue
                yield {"id": item_key, "source": label or f"sel:{name}", "item": item, "row_indices": kept}
    if variants_path:
        for obj in read_jsonl(variants_path):
            key = str(obj.get("id") or obj.get("item_id"))
            if selections and key in selections:
                default_rows = selections[key]["selectors"].get(obj.get("selector", "trimqa_v1"), selections[key]["selectors"].get("trimqa_v1", []))
            else:
                default_rows = obj.get("selected_kept_rows") or []
            for name, item in obj["variants"].items():
                yield {"id": key, "source": f"var:{name}", "item": item, "row_indices": default_rows}


def main():
    p = parse_args()
    p.add_argument("--items", required=True)
    p.add_argument("--selections")
    p.add_argument("--variants")
    p.add_argument("--closed_book", action="store_true")
    p.add_argument("--selectors", action="append", default=[], help="Comma-separated selectors to run, e.g. full,trimqa_v1 or external:gold_only")
    p.add_argument("--model", required=True)
    p.add_argument("--backend", choices=["openai", "mock"], default="openai")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max_tokens", type=int, default=32)
    p.add_argument("--retries", type=int, default=6, help="Retries for transient OpenAI/provider errors")
    p.add_argument("--append", action="store_true", default=True, help="Append to --out when it already exists")
    args = p.parse_args()

    items = load_jsonl_by_id(args.items)
    selections = load_jsonl_by_id(args.selections) if args.selections else {}
    selector_filters = parse_selector_filters(args.selectors)
    jobs = list(make_jobs(items, selections, args.variants, args.closed_book, selector_filters))
    done = load_done(args.out)
    jobs = [job for job in jobs if (job["id"], args.model, job["source"]) not in done]

    client = None
    if args.backend == "openai":
        from openai import OpenAI

        client = OpenAI()

    def run(job):
        item = job["item"]
        table = "" if job["source"] == "closed_book" else table_text(item, job["row_indices"])
        if args.backend == "mock":
            answer = mock_answer(item["question"], table, item["answer"])
        else:
            answer = openai_answer(client, args.model, item["question"], table, args.max_tokens, args.retries)
        correct = canonical_answer(answer) == canonical_answer(item["answer"])
        return {
            "id": job["id"],
            "item_id": job["id"],
            "model": args.model,
            "source": job["source"],
            "condition": job["source"],
            "answer": answer,
            "gold": item["answer"],
            "correct": correct,
            "closed_book": job["source"] == "closed_book",
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append and out.exists() else "w"
    with ThreadPoolExecutor(max_workers=args.workers) as ex, open(out, mode, encoding="utf-8") as fh:
        futures = [ex.submit(run, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="generating"):
            row = fut.result()
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
