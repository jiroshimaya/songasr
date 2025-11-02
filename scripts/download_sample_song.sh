#!/bin/bash
# download_sample_song.sh
# サンプル楽曲をダウンロードして音声認識用にフォーマット変換するスクリプト
# 音源: Pixabay提供のフリー音楽素材

URL="https://cdn.pixabay.com/download/audio/2025/03/22/audio_f46c5fa5ad.mp3"
OUT_MP3="local/song.mp3"
OUT_WAV="local/song_16k.wav"

mkdir -p local
# 1. ダウンロード
curl -L "$URL" -o "$OUT_MP3"

# 2. ffmpegで16kHzモノラルWAVに変換
ffmpeg -i "$OUT_MP3" -ac 1 -ar 16000 -sample_fmt s16 "$OUT_WAV"

echo "変換完了: $OUT_WAV"
