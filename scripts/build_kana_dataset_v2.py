#!/usr/bin/env python3
"""疑似ラベル方式データセット v2: 約1000文 x 複数アクセントでスケールアップ。

v1 (build_kana_dataset.py, 120文・単一アクセント)からの変更点:
- 文の多様性: kana_dataset_sentences_v2.py (手書きシード+複数テンプレート)
- アクセント多様性: gTTSの`tld`パラメータを複数ローテーションし、
  話者/アクセントの偏りを減らす(gTTS自体は話者を選べないため、
  安価にバリエーションを出す手段としてtldを使う)。

音声ファイルは大量になるため、gitにはコミットしない
(local/ 以下はgitignore対象)。再現性のため、マニフェスト
(文・使用tld・疑似ラベルカナ・音声ファイル名)はコミットする。

使用例:
    uv run scripts/build_kana_dataset_v2.py --n 1000 --out local/kana_dataset_v2
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from gtts import gTTS

from kana_dataset_sentences_v2 import build_corpus
from songasr.kana_baseline import baseline_transcribe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_SR = 16_000

# gTTSのtldパラメータでアクセント違いのレンダリングを得る(話者そのものは
# 同じTTSエンジンだが、地域アクセントの発音バリエーションが付く)。
TLD_VARIANTS = ["com", "co.uk", "com.au", "co.in", "ca"]


def synthesize(text: str, tld: str, mp3_path: Path) -> None:
    if mp3_path.exists():
        return
    tts = gTTS(text=text, lang="en", tld=tld)
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


def build(n: int, out_dir: Path, manifest_path: Path) -> None:
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    sentences = build_corpus(target_total=n)
    logger.info("corpus size: %d sentences, tld variants: %s", len(sentences), TLD_VARIANTS)

    tld_cycle = itertools.cycle(TLD_VARIANTS)
    n_empty = 0
    n_ok = 0
    with manifest_path.open("w", encoding="utf-8") as f:
        for i, text in enumerate(sentences, 1):
            uid = f"{i:04d}"
            tld = next(tld_cycle)
            mp3_path = audio_dir / f"{uid}_{tld.replace('.', '-')}.mp3"
            wav_path = audio_dir / f"{uid}_{tld.replace('.', '-')}.wav"
            try:
                synthesize(text, tld, mp3_path)
                audio = load_16k_mono(mp3_path)
                sf.write(str(wav_path), audio, TARGET_SR, subtype="PCM_16")

                result = baseline_transcribe(audio, TARGET_SR)
                if not result["kana"]:
                    n_empty += 1
                else:
                    n_ok += 1
                record = {
                    "id": uid,
                    "text": text,
                    "tld": tld,
                    "audio_file": wav_path.name,
                    "duration_sec": round(len(audio) / TARGET_SR, 3),
                    "kana": result["kana"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                if i % 25 == 0 or i <= 5:
                    logger.info("[%d/%d] (%s) %s -> %s", i, len(sentences), tld, text, result["kana"])
            except Exception:
                logger.exception("failed on sentence %d: %s", i, text)

    logger.info("done. manifest: %s (ok=%d, empty=%d)", manifest_path, n_ok, n_empty)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="生成する文の数")
    parser.add_argument(
        "--out", type=Path, default=Path("local/kana_dataset_v2"),
        help="音声の出力先(gitignore対象)",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/kana_dataset_v2/manifest.jsonl"),
        help="マニフェストの出力先(gitコミット対象)",
    )
    args = parser.parse_args()
    build(args.n, args.out, args.manifest)


if __name__ == "__main__":
    main()
