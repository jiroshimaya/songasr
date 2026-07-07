#!/usr/bin/env python3
"""exp6 結果の集計レポート (summary.json / results.jsonl から)。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "local/exp6"

s = json.load(open(OUT / "summary.json"))

print("== overall ==")
for r in s["overall"]:
    print(f"{r['method']:20s} scored {r['n_scored']:3d}/653  mean {r['mean_dist']:7.2f}  D/mora {r['d_per_mora']:.3f}")
print()
print("== head-to-head vs baseline_mix ==")
for m, h in s["head_to_head"].items():
    print(f"{m:20s} win {h['win_rate']:.1%} (n={h['n']})")
print()

keys = ["baseline_mix", "finetuned_mix", "e2k_english_sung", "whisper_song_e2k",
        "e2k_english_ref", "baseline_vocals"]
print("== per-song D/mora (scored) ==")
print("song".ljust(22), *[k[:15].rjust(16) for k in keys])
for song, ms in s["per_song"].items():
    d = {m["method"]: m for m in ms}
    row = []
    for k in keys:
        v = d.get(k)
        row.append((f"{v['d_per_mora']:.2f}({v['n_scored']})" if v else "-").rjust(16))
    print(song[:22].ljust(22), *row)
print()

rows = [json.loads(x) for x in (OUT / "results.jsonl").read_text().splitlines()]

# チャンス水準: 各goldに(別の曲の)無関係なgoldを予測として与えたときの D/mora
import random  # noqa: E402

random.seed(0)
try:
    import re

    from kanasim import create_kana_distance_calculator

    calc = create_kana_distance_calculator()

    def _norm(s):
        return re.sub(r"\s+", "", s or "")

    golds = [(r["gold"], r["gold_mora"], r["song"]) for r in rows if r["gold_mora"] > 0]
    vals = []
    for g, m, song in golds[:300]:
        others = [x for x in golds if x[2] != song]
        og = random.choice(others)[0]
        try:
            vals.append(calc.calculate(_norm(og), _norm(g)) / m)
        except KeyError:
            pass
    print(f"== chance level: D/mora {sum(vals)/len(vals):.3f} (n={len(vals)}) ==")
    print()
except Exception as e:  # noqa: BLE001
    print("chance calc failed:", e)
print("== examples (first clip of selected songs) ==")
for want in ["eraser_0001", "this_is_me_0005", "stand_by_me_0003", "the_greatest_show_0010"]:
    for r in rows:
        if r["id"] == want:
            print(f"--- {want} (gold: {r['gold']})")
            for k in ["baseline_mix", "finetuned_mix", "e2k_english_sung",
                      "whisper_song_e2k", "e2k_english_ref"]:
                p = r.get(f"{k}_pred")
                dd = r.get(f"{k}_dist")
                dm = round(dd / r["gold_mora"], 2) if dd is not None and r["gold_mora"] else None
                print(f"  {k:20s} {dm!s:>6}  {p}")
            if "whisper_song_en" in r:
                print(f"  whisper_song_en:     {r['whisper_song_en']}")
            break
