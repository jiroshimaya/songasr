#!/usr/bin/env python3
"""issue #7: large-v3全曲転写のセグメント単位評価 + 空耳縮約の評価。

ステージ:
  transcribe : 18曲を large-v3 で全曲転写しセグメント(start/end/text)を保存 (GPU)
  eval       : セグメントを yougaku 行にマッチ(>=0.6)し、e2k(+縮約)で採点 (CPU)

評価プロトコルは english_sung(whisper-small, D/mora 5.49) と同型:
ASR自身のセグメントを単位とし、正解カナはファジーマッチした yougaku 行。
train/test 分割は曲単位 (sortしてidx%3==2がtest、6曲)。

使い方 (リポジトリルートで):
  PYTHONPATH=src:scripts:scripts/exp7 .venv/bin/python scripts/exp7/run_exp7.py <stage>
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = ROOT / "local/gold_stage"
OUT = ROOT / "local/exp7"

SMALL_KANA = set("ャュョァィゥェォヮ")
MIN_MATCH = 0.6


def load_songs() -> list[dict]:
    return [json.loads(x)
            for x in (STAGE_DIR / "stage.jsonl").read_text().splitlines() if x.strip()]


def test_slugs(songs: list[dict]) -> set[str]:
    slugs = sorted(s["slug"] for s in songs)
    return {s for i, s in enumerate(slugs) if i % 3 == 2}


def norm_kana(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def mora_len(kana: str) -> int:
    return sum(1 for ch in norm_kana(kana) if ch not in SMALL_KANA)


def run_transcribe() -> None:
    from faster_whisper import WhisperModel

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "segments.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(x)["slug"] for x in out_path.read_text().splitlines()}

    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    for sng in load_songs():
        if sng["slug"] in done:
            continue
        segments, _info = model.transcribe(
            str(STAGE_DIR / sng["wav"]),
            language="en", vad_filter=False, beam_size=5,
        )
        segs = [{"s": round(float(x.start), 3), "e": round(float(x.end), 3),
                 "text": x.text.strip()} for x in segments]
        with out_path.open("a") as f:
            f.write(json.dumps({"slug": sng["slug"], "segments": segs},
                               ensure_ascii=False) + "\n")
        print(f"[transcribe] {sng['slug']}: {len(segs)} segments", flush=True)
    print("[transcribe] done", flush=True)


def english_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z']+", (text or "").lower()) if w.strip("'")]


def make_e2k():
    from e2k import C2K

    c2k = C2K()

    def convert(words: list[str]) -> list[str]:
        out = []
        for w in words:
            w2 = w.replace("'", "")
            if not w2:
                continue
            try:
                out.append(c2k(w2))
            except Exception:  # noqa: BLE001
                out.append("")
        return out

    return convert


def run_eval(use_contract: bool, source: str) -> None:
    """source: 'large' (exp7 segments) or 'sung' (manifest english_sung)."""
    from kanasim import create_kana_distance_calculator

    sys.path.insert(0, str(ROOT / "scripts"))
    from align_song_kana import best_match, load_yougaku_lines  # noqa: E402

    contract = None
    if use_contract:
        from soramimi_contract import contract_words

        contract = contract_words

    calc = create_kana_distance_calculator()
    e2k = make_e2k()
    songs = load_songs()
    tests = test_slugs(songs)

    pairs = []  # (slug, is_test, en_text, gold_kana)
    coverage = {}
    if source == "large":
        seg_map = {json.loads(x)["slug"]: json.loads(x)["segments"]
                   for x in (OUT / "segments.jsonl").read_text().splitlines()}
        for sng in songs:
            lines = load_yougaku_lines(STAGE_DIR / sng["yougaku"])
            matched_lines = set()
            n_match = 0
            for seg in seg_map[sng["slug"]]:
                idx, score = best_match(seg["text"], lines)
                if idx < 0 or score < MIN_MATCH:
                    continue
                n_match += 1
                matched_lines.add(idx)
                pairs.append((sng["slug"], sng["slug"] in tests,
                              seg["text"], lines[idx][1]))
            coverage[sng["slug"]] = {
                "segments_matched": n_match,
                "lines_matched": len(matched_lines), "lines_total": len(lines)}
    elif source == "sung":
        manifest = [json.loads(x) for x in
                    (ROOT / "local/gold_dataset/manifest.jsonl").read_text().splitlines()]
        for row in manifest:
            slug = row["id"].rsplit("_", 1)[0]
            pairs.append((slug, slug in tests, row["english_sung"], row["kana"]))
    elif source == "ref":
        manifest = [json.loads(x) for x in
                    (ROOT / "local/gold_dataset/manifest.jsonl").read_text().splitlines()]
        for row in manifest:
            slug = row["id"].rsplit("_", 1)[0]
            pairs.append((slug, slug in tests, row["english_ref"], row["kana"]))
    else:
        raise ValueError(source)

    rows = []
    for slug, is_test, en, gold in pairs:
        words = english_words(en)
        kanas = e2k(words)
        if contract is not None:
            kanas = contract(words, kanas)
        pred = norm_kana("".join(kanas))
        m = mora_len(gold)
        d = None
        if pred and m > 0:
            try:
                d = calc.calculate(pred, norm_kana(gold)) / m
            except KeyError:
                d = None
        rows.append({"slug": slug, "test": is_test, "en": en, "gold": gold,
                     "pred": pred, "d_per_mora": d})

    def agg(rs):
        vals = [r["d_per_mora"] for r in rs if r["d_per_mora"] is not None]
        if not vals:
            return "n=0"
        return (f"D/mora mean {statistics.mean(vals):.3f} "
                f"median {statistics.median(vals):.3f} (n={len(vals)}/{len(rs)})")

    tag = f"{source}{'+contract' if use_contract else ''}"
    print(f"== {tag} ==")
    print("  all  :", agg(rows))
    print("  train:", agg([r for r in rows if not r["test"]]))
    print("  test :", agg([r for r in rows if r["test"]]))
    if coverage:
        seg_total = sum(c["segments_matched"] for c in coverage.values())
        lm = sum(c["lines_matched"] for c in coverage.values())
        lt = sum(c["lines_total"] for c in coverage.values())
        print(f"  coverage: matched segments {seg_total}, "
              f"yougaku lines {lm}/{lt} ({lm/lt:.1%})")
    out_file = OUT / f"eval_{tag}.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {out_file}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["transcribe", "eval"])
    ap.add_argument("--source", default="large", choices=["large", "sung", "ref"])
    ap.add_argument("--contract", action="store_true")
    args = ap.parse_args()
    if args.stage == "transcribe":
        run_transcribe()
    else:
        run_eval(args.contract, args.source)


if __name__ == "__main__":
    main()
