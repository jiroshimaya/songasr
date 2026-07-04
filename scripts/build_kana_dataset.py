#!/usr/bin/env python3
"""疑似ラベル方式で(音声, カナ)ペアの学習データ候補を作る。

英語文をgTTSで読み上げ音声にし、baselineパイプライン
(wav2vec2-espeak IPA → ARPABET → arpakana, src/songasr/kana_baseline.py)の
出力を「疑似正解カナ」としてペアにする。

注意: これは弱教師あり(weak supervision)。ラベルの質はbaseline自体の精度に
依存しており、baselineが間違えればラベルも間違う。人手で正解を付けた
データではないことに注意。

使用例:
    uv run scripts/build_kana_dataset.py --n 120 --out local/kana_dataset
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from gtts import gTTS

from kana_dataset_sentences import build_corpus

from songasr.kana_baseline import baseline_transcribe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_SR = 16_000


def synthesize(text: str, mp3_path: Path) -> None:
    if mp3_path.exists():
        return
    tts = gTTS(text=text, lang="en")
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    mp3_path.write_bytes(buf.getvalue())


def load_16k_mono(mp3_path: Path, pad_sec: float = 0.3) -> np.ndarray:
    audio, sr = sf.read(str(mp3_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    pad = int(pad_sec * TARGET_SR)
    return np.pad(audio, pad).astype(np.float32)


def build(n: int, out_dir: Path) -> None:
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    sentences = build_corpus(target_total=n)
    logger.info("corpus size: %d sentences", len(sentences))

    n_empty = 0
    with manifest_path.open("w", encoding="utf-8") as f:
        for i, text in enumerate(sentences, 1):
            uid = f"{i:04d}"
            mp3_path = audio_dir / f"{uid}.mp3"
            wav_path = audio_dir / f"{uid}.wav"
            try:
                synthesize(text, mp3_path)
                audio = load_16k_mono(mp3_path)
                sf.write(str(wav_path), audio, TARGET_SR, subtype="PCM_16")

                result = baseline_transcribe(audio, TARGET_SR)
                if not result["kana"]:
                    n_empty += 1
                record = {
                    "id": uid,
                    "text": text,
                    "audio": str(wav_path.relative_to(out_dir)),
                    "duration_sec": round(len(audio) / TARGET_SR, 3),
                    "ipa": result["ipa"],
                    "arpabet": result["arpabet"],
                    "kana": result["kana"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                logger.info("[%d/%d] %s -> %s", i, len(sentences), text, result["kana"])
            except Exception:
                logger.exception("failed on sentence %d: %s", i, text)

    logger.info("done. manifest: %s (empty kana labels: %d)", manifest_path, n_empty)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=120, help="生成する文の数")
    parser.add_argument(
        "--out", type=Path, default=Path("local/kana_dataset"), help="出力先ディレクトリ"
    )
    args = parser.parse_args()
    build(args.n, args.out)


if __name__ == "__main__":
    main()
