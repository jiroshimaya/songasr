#!/usr/bin/env python3
"""train曲の gold ↔ e2k(english_ref) の差分パターンを頻度集計する (ルール設計用)。"""
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/exp7"))
from run_exp7 import english_words, load_songs, make_e2k, norm_kana, test_slugs  # noqa: E402

songs = load_songs()
tests = test_slugs(songs)
print("test songs:", sorted(tests))
e2k = make_e2k()

manifest = [json.loads(x) for x in
            (ROOT / "local/gold_dataset/manifest.jsonl").read_text().splitlines()]

ops = Counter()
word_pairs = Counter()
for row in manifest:
    slug = row["id"].rsplit("_", 1)[0]
    if slug in tests:
        continue
    words = english_words(row["english_ref"])
    pred = norm_kana("".join(e2k(words)))
    gold = norm_kana(row["kana"])
    if not pred or not gold:
        continue
    sm = difflib.SequenceMatcher(None, pred, gold)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        # 前後1文字の文脈をつけて集計
        src = pred[max(0, i1 - 1):i2 + 1]
        dst = gold[max(0, j1 - 1):j2 + 1]
        ops[(tag, pred[i1:i2], gold[j1:j2])] += 1

print("\n== top 60 diff ops (tag, e2k側, gold側) ==")
for (tag, a, b), c in ops.most_common(60):
    print(f"{c:4d}  {tag:8s} {a!r} -> {b!r}")
