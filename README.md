# What Relevance Pruning Leaves Behind in Table Question Answering: Supplement

This anonymized supplement contains the per-question artifacts, saved model
outputs, audit checks, and stats-only code used for the WikiSQL experiments in
the paper. It does not redistribute WikiSQL tables; use the original WikiSQL
release and the item/table IDs in `data/questions_test.jsonl`.

## Contents

- `data/questions_test.jsonl`: one row per built test question, with WikiSQL item
  ID, table ID, SQL, inclusion status, and label-exclusion reason.
- `data/excluded_13.csv`: the 13 test label exclusions used in Appendix B.
- `data/kept_rows.jsonl`: rows kept by TRIM-QA v1 and by same-budget sim-top-k.
- `data/arms.jsonl`: BASE/TREAT/CTRL row lists, matched status, control pairs,
  score/length gaps, filler rows, filler type, and strict filler condition vector.
- `data/filler_sources.csv`: candidate source rows recovered after the run by
  exact matching against WikiSQL `test.tables.jsonl`; the original intervention
  log did not store exact filler provenance.
- `data/row_labels.jsonl`: gold / competing-near-match / low-match / other row labels by item ID, table ID, and row index. Full WikiSQL row text is not included.
- `outputs/*.jsonl`: raw saved model answers for primary arms, with scorer flags.
- `checks/llm_review_253.csv`: blinded Codex/GPT-5-based review of the 253
  automatically discordant TREAT/CTRL pairs.
- `checks/human_check_50.csv`: deterministic 50-row human-check template; labels
  were not present in the experiment folder when this package was built.
- `checks/wtq_screen.csv`: summary placeholder for the WTQ/SQUALL screen.
- `code/`: scripts corresponding to row selection, arm construction, prompting,
  scoring, and statistics.
- `protocol/protocol_dev.md`: frozen protocol, settings, and script hashes.

## Models and Encoder

- Closed model: `gpt-4o-2024-05-13`
- Open models: `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`
- Row encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Decoding: temperature 0, max tokens 100

## Paper Number Map

| Paper number / table entry | File(s) | How to verify |
|---|---|---|
| Selection audit 76.4% vs 61.6% | `data/kept_rows.jsonl`, `data/row_labels.jsonl` | Run selection audit in `code/stats.py` or aggregate kept competing-near-match vs low-match rows. |
| Primary TREAT-CTRL, matched questions | `outputs/*.jsonl`, `data/arms.jsonl` | `python code/stats.py --items data/questions_test.jsonl --answers outputs/gpt4o_run1.jsonl --variants data/arms.jsonl --out /tmp/report.json` after adapting paths to full item rows, or inspect `checks/reviewer_audit_summary.json`. |
| Primary effects +2.6 / +5.3 / +5.1 | `checks/reviewer_audit_summary.json` | `models[*].primary_treat_vs_ctrl.mean`. |
| CTRL competitor transitions 140 -> 86 | `checks/reviewer_audit_summary.json` | Sum `ctrl_competitor_answer_transitions_to_treat`: CTRL competitor total 23+54+63=140; stayed competitor 0+8+7 plus TREAT competitor from non-CTRL-competitor cases gives the reported residual competitor counts. |
| 460 questions within tolerance | `data/arms.jsonl` | Count primary-matched questions where every control pair is within delta 0.05 and 25% length tolerance. |
| 5 questions with all-condition filler | `checks/reviewer_audit_summary.json`, `data/arms.jsonl` | `corrected_filler_condition_recount.question_counts.any_all_conditions`. |
| 253 discordant review rows | `checks/llm_review_253.csv` | Count rows. |

## Stats-Only Reruns

These reruns use saved outputs only; no API calls or GPUs are needed.

```bash
python code/reviewer_audit.py --items data/questions_test.jsonl \
  --variants data/arms.jsonl --answers outputs/gpt4o_run1.jsonl \
  --out /tmp/reviewer_audit
```

The packaged `code/` scripts are the experiment scripts from the run. Some expect
full WikiSQL item JSON with table rows; `data/questions_test.jsonl` intentionally
contains metadata only to avoid redistributing the dataset. To rerun end-to-end,
first download WikiSQL from the original Salesforce release and rebuild items
with `code/label_rows.py`.

## WikiSQL

Download WikiSQL from the original Salesforce release:
`https://github.com/salesforce/WikiSQL`. This supplement uses IDs and table IDs
rather than redistributing full WikiSQL tables.

## Known Packaging Notes

- `checks/human_check_50.csv` is a template/sample because no completed human
  check file was present when this supplement was built.
- `checks/wtq_screen.csv` is a summary placeholder because no per-question
  WTQ/SQUALL screen file was present locally.
- `data/filler_sources.csv` is recovered after the run and should not be read as
  an original provenance log.
