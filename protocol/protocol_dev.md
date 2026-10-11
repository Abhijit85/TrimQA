# Frozen Protocol Summary

This file records the confirmatory protocol used for the clean WikiSQL test run.
It is derived from `freeze.json` in the experiment folder.

- Frozen at UTC: 2026-10-04T20:57:24.456709+00:00
- Pilot split: dev
- Confirmatory split: test
- Minimum rows: 16
- Primary selector: trimqa_v1
- Primary contrast: var:treat minus var:ctrl, similarity_matched subset, all clean matched items
- Primary outcome: answer_match v2: normalized exact/number match or whole-token gold span with at most 4 extra tokens; includes the A/article-empty fix, dash/minus numeric normalization, WikiSQL display-artifact normalization, and letter-digit boundary date normalization
- Filler source: same_header_tables
- Delta: 0.05
- Length tolerance: 0.25
- Success rule: 95% cluster-bootstrap CI excludes 0 for >=2 models, same sign in the third

## Models

[
  "gpt-4o-2024-05-13",
  "Qwen/Qwen2.5-7B-Instruct",
  "meta-llama/Llama-3.1-8B-Instruct"
]

## Script SHA1 at freeze

```json
{
  "analyze.py": "e4227b8416b7af98e31be40f2b72d0b8b4952f01",
  "answer_metric.py": "a24379a34bd6b5b9a6a8cda44199dfaa01f2fbbe",
  "build_items.py": "1eed4b960803b2c07140f131ced1ba664efad467",
  "freeze.py": "e68fbbb5b3d934d2662c620b4adc82b484c0cf93",
  "generate.py": "30c726d94087c876508773fd1b320b7b388cefca",
  "interventions.py": "adfb87faaba5e9df23891a913c4c43acf9851644",
  "make_gold_only.py": "aac09bffb860513d06a6f33719a42eef9c67c49f",
  "prune.py": "cb8ef8d75b294ebd5dbfb19cdfd9874c166ba380",
  "run_wikisql_dev.sh": "c1ae7ea147721de296706622709d9a1850fdac07",
  "run_wikisql_test.sh": "45cdca566980fd40176c98f866c0d596741455a1",
  "run_wikisql_test_primary.sh": "8144c331819c0395c862a10662536586f30719ab",
  "subset_analyze.py": "e9f35b8eeecb7f94700d0e6137519756de6e9b27",
  "wikisql_trimqa_common.py": "33aba182f88c3e110115d82f09426a7f9c3cb86b"
}
```

## Post-freeze notes

```json
[
  "Post-freeze analysis-only fix on 2026-10-04: analyze.py gold-only-correct subset now falls back from sel:external:gold_only to sel:gold_only because the clean test selector condition is named sel:gold_only. Generated answers, variants, primary metric, prompt, model settings, and primary all-item treat-vs-ctrl contrast are unchanged.",
  "Post-freeze selector audit on 2026-10-04: prune.py oracle_random was fixed to preserve only the support row instead of forcing competing near-match rows. Rerunning test selections after the fix changed oracle_random for 867/984 items but trimqa_v1 and sim_topk were identical to the frozen sel_clean.jsonl (0 diffs each), so primary and sim_topk analyses are unchanged."
]
```
