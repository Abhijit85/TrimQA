#!/usr/bin/env python3
"""Similarity top-k selector used in the supplement.

The actual implementation is in code/trimqa.py: rows are scored with
sentence-transformers/all-MiniLM-L6-v2 and sim_topk keeps the same number of
rows as trimqa_v1 by descending score, ties by row index.
"""
from trimqa import topk
