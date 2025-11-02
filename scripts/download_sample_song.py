#!/usr/bin/env python3
"""download_sample_song.py

サンプル楽曲をダウンロードしてffmpegを使わずに純PythonでWAV変換するスクリプト
音源: Pixabay提供のフリー音楽素材
"""

import logging
from pathlib import Path

import librosa
import requests
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

URL = "https://cdn.pixabay.com/download/audio/2025/03/22/audio_f46c5fa5ad.mp3"
OUT_MP3 = Path("local/song.mp3")
OUT_WAV = Path("local/song_16k_pure.wav")


def download_sample_song() -> None:
    """ffmpegを使わずに純Pythonでサンプル楽曲をダウンロードしてWAV形式に変換する"""
    # ディレクトリを作成
    OUT_MP3.parent.mkdir(exist_ok=True)

    # 1. ダウンロード
    logger.info(f"Downloading from {URL}")
    response = requests.get(URL, stream=True, timeout=30)
    response.raise_for_status()

    with OUT_MP3.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(f"Downloaded: {OUT_MP3}")

    # 2. librosaで音声ファイルを読み込み（自動的にモノラル変換）
    logger.info("Converting to 16kHz mono WAV")

    # librosaはMP3をnumpyのndarrayとして読み込む（自動的にモノラル化）
    # sr=16000で16kHzにリサンプリング
    audio_data, sample_rate = librosa.load(OUT_MP3, sr=16000, mono=True)

    logger.info(f"Loaded audio: shape={audio_data.shape}, sr={sample_rate}")

    # 3. soundfileでWAVとして保存
    sf.write(OUT_WAV, audio_data, sample_rate, subtype="PCM_16")

    logger.info(f"変換完了: {OUT_WAV}")


if __name__ == "__main__":
    download_sample_song()
