#!/usr/bin/env python3
"""Demucsでボーカル分離した音声に対してbaselineパイプラインを実行し、
分離前後でカナ認識がどう変わるかを見る。

kana_asr_experimentの検証で、"Oh eternal light, that shines so pure"の
区間(63.72-69.72秒)がbaselineで"ス"の1文字にまで崩壊していた。これが
伴奏(BGM)の干渉によるものか、それとも歌唱表現(メリスマ等)自体が原因かを
切り分けるための診断。
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from songasr.kana_baseline import baseline_transcribe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_SR = 16_000
SONG_MP3 = Path("local/song.mp3")
SONG_URL = "https://cdn.pixabay.com/download/audio/2025/03/22/audio_f46c5fa5ad.mp3"
SEP_DIR = Path("local/demucs_out")

# kana_asr_experimentで検証済みの区間(前後パディング込みで同じ切り出し方)
SEGMENT = {"start": 63.72, "end": 69.72, "text": "Oh eternal light, that shines so pure"}
PAD_BEFORE, PAD_AFTER = 0.2, 0.3


def ensure_song() -> None:
    if SONG_MP3.exists():
        return
    import requests

    SONG_MP3.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(SONG_URL, timeout=30)
    resp.raise_for_status()
    SONG_MP3.write_bytes(resp.content)
    logger.info("downloaded %s", SONG_MP3)


def run_demucs() -> Path:
    """demucsのCLI (subprocess) はtorchaudio->torchcodecがffmpeg共有ライブラリを
    要求し、この環境では読み込みに失敗するため、Python APIを直接呼んで
    librosa/soundfileで読み込んだ波形をそのままモデルに渡す。
    """
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    SEP_DIR.mkdir(parents=True, exist_ok=True)
    model = get_model("htdemucs")
    model.eval()

    audio, sr = librosa.load(str(SONG_MP3), sr=model.samplerate, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    wav = torch.from_numpy(audio).float()
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()

    with torch.no_grad():
        sources = apply_model(model, wav[None], device="cpu", progress=True)[0]
    sources = sources * ref.std() + ref.mean()

    vocal_idx = model.sources.index("vocals")
    vocals = sources[vocal_idx].numpy()  # shape (channels, samples)

    out_dir = SEP_DIR / "htdemucs" / SONG_MP3.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    vocal_path = out_dir / "vocals.wav"
    sf.write(str(vocal_path), vocals.T, model.samplerate, subtype="PCM_16")
    return vocal_path


def extract_segment_16k(wav_path: Path, start: float, end: float) -> np.ndarray:
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    s = max(0, int((start - PAD_BEFORE) * TARGET_SR))
    e = min(len(audio), int((end + PAD_AFTER) * TARGET_SR))
    return audio[s:e]


def main() -> None:
    ensure_song()
    vocal_path = run_demucs()
    logger.info("separated vocals: %s", vocal_path)

    clip = extract_segment_16k(vocal_path, SEGMENT["start"], SEGMENT["end"])
    sf.write("local/demucs_out/eternal_light_vocals_only.wav", clip, TARGET_SR, subtype="PCM_16")

    result = baseline_transcribe(clip, TARGET_SR)
    logger.info("text: %s", SEGMENT["text"])
    logger.info("baseline (vocals-only) ipa: %s", result["ipa"])
    logger.info("baseline (vocals-only) kana: %s", result["kana"])
    print(f"\n[RESULT] vocals-only baseline kana: {result['kana']!r}")
    print("[COMPARE] original (with BGM) baseline kana from kana_asr_experiment: 'ス'")


if __name__ == "__main__":
    main()
