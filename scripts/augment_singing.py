#!/usr/bin/env python3
"""TTS音声に歌唱っぽい音響変化(ピッチシフト・テンポ変更)を加えて、
実歌唱とのギャップを安く埋められるか試すプロトタイプ。

librosaのpitch_shift/time_stretchを使う。実際に歌唱で起きている
メリスマ(1音素を複数の音高にまたがって伸ばす)やビブラートそのものは
再現できないため、その限界も合わせて記録する。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from songasr.kana_baseline import baseline_transcribe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_SR = 16_000


def augment_variants(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    variants = {"original": audio}
    # ピッチシフト (半音単位。歌唱のメロディに乗る音高変化を模す)
    variants["pitch_up_4st"] = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
    variants["pitch_down_4st"] = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-4)
    # テンポ変更 (歌唱で音を伸ばす/縮めるのを模す)
    variants["stretch_slow_0.7x"] = librosa.effects.time_stretch(audio, rate=0.7)
    variants["stretch_fast_1.3x"] = librosa.effects.time_stretch(audio, rate=1.3)
    return variants


def spectral_stats(audio: np.ndarray, sr: int) -> dict[str, float]:
    """耳で聴けない代わりに目で見て健全性を確認するための簡易統計。"""
    duration = len(audio) / sr
    rms = float(np.sqrt(np.mean(audio**2)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
    )
    voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
    mean_f0 = float(np.nanmean(voiced)) if len(voiced) else float("nan")
    return {
        "duration_sec": round(duration, 3),
        "rms": round(rms, 5),
        "spectral_centroid_hz": round(centroid, 1),
        "mean_f0_hz": round(mean_f0, 1) if mean_f0 == mean_f0 else None,  # NaN check
    }


def main() -> None:
    audio_dir = Path("local/augment_demo")
    audio_dir.mkdir(parents=True, exist_ok=True)

    # kana_asr_experimentで既に作ったTTS音声を再利用(なければ都度合成でも可)
    src_wav = Path(
        "/tmp/claude-1000/-home-jiro-development-soramimic/"
        "da84d668-b59b-4141-8a5e-d4b956b107ae/scratchpad/kana_asr_experiment/"
        "audio/s1_moon.wav"
    )
    audio, sr = sf.read(str(src_wav), dtype="float32")
    assert sr == TARGET_SR

    results = {}
    for name, variant in augment_variants(audio, sr).items():
        out_path = audio_dir / f"{name}.wav"
        sf.write(str(out_path), variant, sr, subtype="PCM_16")
        stats = spectral_stats(variant, sr)
        asr = baseline_transcribe(variant, sr)
        results[name] = {"stats": stats, "kana": asr["kana"]}
        logger.info("%s: stats=%s kana=%s", name, stats, asr["kana"])

    Path("local/augment_demo/results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
