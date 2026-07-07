#!/usr/bin/env python3
"""whisper_song(midpoint割当) と e2k_english_sung のペア比較。"""
import json
import re
import statistics
from pathlib import Path

from e2k import C2K
from kanasim import create_kana_distance_calculator

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "local/exp6"
GOLD = ROOT / "local/gold_dataset"
c2k = C2K()
calc = create_kana_distance_calculator()
SMALL = set("ャュョァィゥェォヮ")


def norm(s):
    return re.sub(r"\s+", "", s or "")


def mora(s):
    return sum(1 for ch in norm(s) if ch not in SMALL)


def to_kana(text):
    out = []
    for w in re.findall(r"[a-zA-Z']+", (text or "").lower()):
        w = w.replace("'", "")
        if w:
            try:
                out.append(c2k(w))
            except Exception:
                pass
    return "".join(out)


def d(pred_kana, gold, m):
    if not norm(pred_kana) or m == 0:
        return None
    try:
        return calc.calculate(norm(pred_kana), norm(gold)) / m
    except KeyError:
        return None


song_words = {json.loads(x)["id"]: json.loads(x)["words"]
              for x in (OUT / "whisper_song.jsonl").read_text().splitlines()}
manifest = [json.loads(x) for x in (GOLD / "manifest.jsonl").read_text().splitlines()]

pairs = []
for row in manifest:
    slug = row["id"].rsplit("_", 1)[0]
    if slug not in song_words:
        continue
    s, e = row["start_sec"], row["end_sec"]
    txt = "".join(w["w"] for w in song_words[slug]
                  if s <= (w["s"] + w["e"]) / 2 <= e)
    m = mora(row["kana"])
    a = d(to_kana(txt), row["kana"], m)
    b = d(to_kana(row["english_sung"]), row["kana"], m)
    if a is not None and b is not None:
        pairs.append((a, b, row["id"]))

am = [p[0] for p in pairs]
bm = [p[1] for p in pairs]
print(f"n={len(pairs)}")
print(f"whisper_song(midpoint): mean {statistics.mean(am):.3f} median {statistics.median(am):.3f}")
print(f"e2k_english_sung:       mean {statistics.mean(bm):.3f} median {statistics.median(bm):.3f}")
wins = sum(1 for a, b, _ in pairs if a < b)
print(f"whisper_song wins: {wins}/{len(pairs)} ({wins/len(pairs):.1%})")
# 大差で負けるケースの分布
diffs = sorted((a - b, cid) for a, b, cid in pairs)
big = [x for x in diffs if x[0] > 2]
print(f"large regressions (diff>2 D/mora): {len(big)}")
for x in big[-5:]:
    print("  ", x[1], round(x[0], 2))
