#!/usr/bin/env python3
"""購入した実歌唱音源と、yougaku由来の(英語歌詞, カナ)行対応データを突き合わせ、
(音声区間, カナ)ペアを自動生成するパイロット。

処理:
  1. wav(16kHz mono)をfaster-whisperで単語タイムスタンプ付き書き起こし → whisper区間を得る
  2. 各whisper区間の英語テキストを、yougakuの英語行にファジーマッチし、対応するカナ行を付与
  3. 各区間を切り出してbaseline(疑似ラベラー)で認識し、yougakuカナをリファレンスにkanasim距離を計算
     (--finetuned 指定時は学習済みLoRAモデルでも認識・比較)

出力: 区間ごとの {start,end, whisper_en, matched_yougaku_en, match_score, kana(ref), baseline_kana, baseline_dist, ...}

使い方:
  uv run python scripts/align_song_kana.py \
      --audio local/yougaku/stand_by_me_16k.wav \
      --yougaku local/yougaku/stand_by_me.json \
      --out local/yougaku/stand_by_me_aligned.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from kanasim import create_kana_distance_calculator

from songasr.kana_baseline import baseline_transcribe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("align_song_kana")

TARGET_SR = 16_000


def norm_en(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_yougaku_lines(path: Path) -> list[tuple[str, str]]:
    """yougakuレコードから (英語行, カナ行) の対応リストを返す。"""
    d = json.loads(path.read_text(encoding="utf-8"))
    en = [x for x in d["abc"].split("/") if x.strip()]
    ka = [x for x in d["kana"].split("/") if x.strip()]
    n = min(len(en), len(ka))
    return list(zip(en[:n], ka[:n], strict=False))


def best_match(text: str, lines: list[tuple[str, str]]) -> tuple[int, float]:
    """whisper区間テキストに最も近いyougaku英語行のindexとスコアを返す。"""
    nt = norm_en(text)
    best_i, best_r = -1, 0.0
    for i, (en, _ka) in enumerate(lines):
        r = difflib.SequenceMatcher(None, nt, norm_en(en)).ratio()
        if r > best_r:
            best_i, best_r = i, r
    return best_i, best_r


def transcribe_whisper(wav_path: Path, model_size: str) -> list[dict[str, Any]]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    # vad_filterは話し声向けで、歌唱を非音声として全除去してしまうため無効化
    segments, info = model.transcribe(
        str(wav_path), language="en", word_timestamps=True, vad_filter=False
    )
    logger.info("whisper言語=%s 確率=%.2f", info.language, info.language_probability)
    segs = []
    for s in segments:
        segs.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip()})
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--yougaku", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--whisper-size", default="large-v3")
    ap.add_argument("--min-score", type=float, default=0.4, help="この類似度未満のマッチは低信頼として印")
    ap.add_argument("--finetuned", type=Path, default=None, help="学習済みLoRAアダプタのパス(任意)")
    args = ap.parse_args()

    audio, sr = sf.read(str(args.audio), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    lines = load_yougaku_lines(args.yougaku)
    logger.info("yougaku行数=%d", len(lines))

    segs = transcribe_whisper(args.audio, args.whisper_size)
    logger.info("whisper区間数=%d", len(segs))

    calc = create_kana_distance_calculator()

    ft = None
    if args.finetuned:
        ft = _load_finetuned(args.finetuned)

    records = []
    for i, seg in enumerate(segs, 1):
        idx, score = best_match(seg["text"], lines)
        ref_en, ref_kana = lines[idx] if idx >= 0 else ("", "")
        s = max(0, int(seg["start"] * TARGET_SR))
        e = min(len(audio), int(seg["end"] * TARGET_SR))
        clip = audio[s:e]
        base = baseline_transcribe(clip, TARGET_SR)
        base_dist = _safe_dist(calc, base["kana"], ref_kana)
        rec = {
            "index": i, "start": seg["start"], "end": seg["end"],
            "whisper_en": seg["text"], "matched_yougaku_en": ref_en,
            "match_score": round(score, 3), "low_confidence": score < args.min_score,
            "kana_ref": ref_kana,
            "baseline_kana": base["kana"], "baseline_dist": base_dist,
        }
        if ft is not None:
            ft_kana = _ft_transcribe(ft, clip)
            rec["finetuned_kana"] = ft_kana
            rec["finetuned_dist"] = _safe_dist(calc, ft_kana, ref_kana)
        records.append(rec)
        logger.info("[%02d] score=%.2f en=%r ref_kana=%s base=%s(%s)",
                    i, score, seg["text"][:40], ref_kana, base["kana"], base_dist)

    args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    good = [r for r in records if not r["low_confidence"]]
    bd = [r["baseline_dist"] for r in good if r["baseline_dist"] is not None]
    logger.info("=== まとめ ===")
    logger.info("区間 %d 件 / 高信頼マッチ %d 件 (score>=%.2f)", len(records), len(good), args.min_score)
    if bd:
        logger.info("高信頼区間の baseline 平均kanasim距離: %.2f (n=%d)", sum(bd) / len(bd), len(bd))
    if ft is not None:
        fd = [r["finetuned_dist"] for r in good if r.get("finetuned_dist") is not None]
        if fd:
            logger.info("高信頼区間の finetuned 平均kanasim距離: %.2f (n=%d)", sum(fd) / len(fd), len(fd))
    logger.info("出力: %s", args.out)


def _safe_dist(calc: Any, kana: str, ref: str) -> float | None:
    # kanasim はカナ内の半角スペースで (モーラ, ' ') の遷移コスト参照に失敗し
    # KeyError を投げる。スペースは音韻情報を持たないため採点前に除去する
    # (これを怠ると採点不能が多発し、認識崩壊と区別できなくなる)。
    if kana:
        kana = re.sub(r"\s+", "", kana)
    if ref:
        ref = re.sub(r"\s+", "", ref)
    if not kana or not ref:
        return None
    try:
        return calc.calculate(kana, ref)
    except KeyError:
        return None


def _load_finetuned(adapter_path: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    from songasr.kana_wav2vec2_ja import MODEL_ID

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval().to("cuda")
    return {"model": model, "processor": processor}


def _ft_transcribe(ft: dict[str, Any], clip: np.ndarray) -> str:
    import torch

    from songasr.kana_wav2vec2_ja import text_to_katakana

    proc = ft["processor"]
    # 極端に短い区間はwav2vec2のconv層(最小受容野)を下回りクラッシュするためゼロパディング
    if len(clip) < 8000:
        clip = np.pad(clip, (0, 8000 - len(clip)))
    inputs = proc(clip, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        logits = ft["model"](inputs.input_values.to("cuda")).logits
    ids = torch.argmax(logits, dim=-1)
    raw = proc.batch_decode(ids)[0]
    return text_to_katakana(raw)


if __name__ == "__main__":
    main()
