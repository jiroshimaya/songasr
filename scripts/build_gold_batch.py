#!/usr/bin/env python3
"""GPU側: ステージした複数曲を一括で処理し、gold データセットを構築する。

whisper と学習済みモデルを1回だけロードして stage.jsonl の全曲をループ処理:
  wav -> whisper区間 -> yougaku英語行にファジーマッチ -> カナ付与 -> 区間切り出し
  -> baseline / finetuned で認識 -> kanasim距離。 高信頼(score>=min-score)のみ gold採用。

出力: <outdir>/{clips/, manifest.jsonl, summary.json}

使い方(要 LD_LIBRARY_PATH で cudnn/cublas を通す):
  uv run python scripts/build_gold_batch.py --stage local/gold_stage \
      --outdir local/gold_dataset --finetuned local/models/kana_ctc_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from kanasim import create_kana_distance_calculator

from songasr.kana_baseline import baseline_transcribe

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_song_kana import (  # noqa: E402
    TARGET_SR,
    _ft_transcribe,
    _load_finetuned,
    _safe_dist,
    best_match,
    load_yougaku_lines,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_gold_batch")


def load_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--finetuned", type=Path, default=None,
                    help="学習済みLoRA(任意)。未指定なら採点をスキップ(GPU不使用)")
    ap.add_argument("--whisper-size", default="large-v3")
    ap.add_argument("--whisper-device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--min-score", type=float, default=0.6)
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    compute = "float16" if args.whisper_device == "cuda" else "int8"
    logger.info("whisper(%s) を %s(%s) でロード中...", args.whisper_size, args.whisper_device, compute)
    wmodel = WhisperModel(args.whisper_size, device=args.whisper_device, compute_type=compute)
    ft = None
    if args.finetuned:
        logger.info("学習済みモデル ロード中...")
        ft = _load_finetuned(args.finetuned)
    calc = create_kana_distance_calculator()

    clips_dir = args.outdir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    songs = [json.loads(x) for x in (args.stage / "stage.jsonl").read_text().splitlines() if x.strip()]
    logger.info("処理対象: %d 曲", len(songs))

    all_rows: list[dict] = []
    per_song = []
    for si, sng in enumerate(songs, 1):
        wav = load_wav(args.stage / sng["wav"])
        lines = load_yougaku_lines(args.stage / sng["yougaku"])
        segments, info = wmodel.transcribe(
            str(args.stage / sng["wav"]), language="en",
            word_timestamps=True, vad_filter=False,
        )
        rows = []
        kept = 0
        for seg in segments:
            idx, score = best_match(seg.text.strip(), lines)
            if score < args.min_score:
                continue
            ref_en, ref_kana = lines[idx]
            s = max(0, int(seg.start * TARGET_SR))
            e = min(len(wav), int(seg.end * TARGET_SR))
            clip = wav[s:e]
            if len(clip) < 800:  # ほぼ空の区間は捨てる
                continue
            kept += 1
            cid = f"{sng['slug']}_{kept:04d}"
            sf.write(str(clips_dir / f"{cid}.wav"), clip, TARGET_SR, subtype="PCM_16")
            base = baseline_transcribe(clip, TARGET_SR)
            ftk = _ft_transcribe(ft, clip) if ft is not None else None
            rows.append({
                "id": cid, "audio_file": f"clips/{cid}.wav",
                "song": sng["song"], "recording": sng["recording"],
                "start_sec": round(float(seg.start), 3), "end_sec": round(float(seg.end), 3),
                "duration_sec": round((e - s) / TARGET_SR, 3),
                "english_sung": seg.text.strip(), "english_ref": ref_en,
                "kana": ref_kana, "kana_source": "yougaku-nihongo.com",
                "match_score": round(score, 3),
                "baseline_pred": base["kana"], "baseline_kanasim": _safe_dist(calc, base["kana"], ref_kana),
                "finetuned_pred": ftk, "finetuned_kanasim": _safe_dist(calc, ftk, ref_kana),
            })
        all_rows.extend(rows)
        bd = [r["baseline_kanasim"] for r in rows if r["baseline_kanasim"] is not None]
        fd = [r["finetuned_kanasim"] for r in rows if r["finetuned_kanasim"] is not None]
        stat = {
            "song": sng["song"], "recording": sng["recording"], "clips": len(rows),
            "baseline_scored": len(bd), "baseline_mean": round(sum(bd) / len(bd), 1) if bd else None,
            "finetuned_scored": len(fd), "finetuned_mean": round(sum(fd) / len(fd), 1) if fd else None,
        }
        per_song.append(stat)
        logger.info("[%02d/%d] %-22s clips=%d base=%s(n%d) ft=%s(n%d)",
                    si, len(songs), sng["song"][:22], len(rows),
                    stat["baseline_mean"], len(bd), stat["finetuned_mean"], len(fd))

    with (args.outdir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 全体サマリ(両モデル採点可の同一区間ペア比較)
    both = [r for r in all_rows if r["baseline_kanasim"] is not None and r["finetuned_kanasim"] is not None]
    summary = {
        "total_clips": len(all_rows),
        "baseline_scored": sum(1 for r in all_rows if r["baseline_kanasim"] is not None),
        "finetuned_scored": sum(1 for r in all_rows if r["finetuned_kanasim"] is not None),
        "paired_n": len(both),
        "paired_baseline_mean": round(sum(r["baseline_kanasim"] for r in both) / len(both), 1) if both else None,
        "paired_finetuned_mean": round(sum(r["finetuned_kanasim"] for r in both) / len(both), 1) if both else None,
        "finetuned_wins": sum(1 for r in both if r["finetuned_kanasim"] < r["baseline_kanasim"]),
        "per_song": per_song,
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("=== 完了 ===")
    logger.info("総クリップ: %d (baseline採点可 %d / finetuned採点可 %d)",
                summary["total_clips"], summary["baseline_scored"], summary["finetuned_scored"])
    logger.info("同一区間ペア(%d): baseline平均=%s finetuned平均=%s finetuned勝ち=%d",
                summary["paired_n"], summary["paired_baseline_mean"],
                summary["paired_finetuned_mean"], summary["finetuned_wins"])
    logger.info("出力: %s", args.outdir)


if __name__ == "__main__":
    main()
