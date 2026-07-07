#!/usr/bin/env python3
"""whisper_song の単語→クリップ割当パディングの感度分析。"""
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


song_words = {}
for line in (OUT / "whisper_song.jsonl").read_text().splitlines():
    r = json.loads(line)
    song_words[r["id"]] = r["words"]

manifest = [json.loads(x) for x in (GOLD / "manifest.jsonl").read_text().splitlines()]


def pick(words, s, e, mode, pad):
    if mode == "overlap":
        return "".join(w["w"] for w in words if w["e"] > s - pad and w["s"] < e + pad)
    if mode == "midpoint":
        return "".join(w["w"] for w in words if s - pad <= (w["s"] + w["e"]) / 2 <= e + pad)
    if mode == "strict":
        return "".join(w["w"] for w in words if w["s"] >= s - pad and w["e"] <= e + pad)
    raise ValueError(mode)


for mode, pad in [("overlap", 0.2), ("overlap", 0.0), ("midpoint", 0.0),
                  ("strict", 0.2), ("strict", 0.5)]:
    vals = []
    n_empty = 0
    for row in manifest:
        slug = row["id"].rsplit("_", 1)[0]
        if slug not in song_words:
            continue
        txt = pick(song_words[slug], row["start_sec"], row["end_sec"], mode, pad)
        kana = to_kana(txt)
        m = mora(row["kana"])
        if not norm(kana) or m == 0:
            n_empty += 1
            continue
        try:
            vals.append(calc.calculate(norm(kana), norm(row["kana"])) / m)
        except KeyError:
            pass
    print(f"{mode:9s} pad={pad:4.1f}  D/mora {statistics.mean(vals):.3f} "
          f"(scored {len(vals)}, empty {n_empty})")
