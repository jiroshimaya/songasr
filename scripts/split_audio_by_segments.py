#!/usr/bin/env python3
"""
音声ファイルをWhisperの文字起こし結果のセグメントに基づいて分割するスクリプト

使用例:
    uv run scripts/split_audio_by_segments.py \
        --whisper-json local/whisper_transcription.json \
        --wav-file path/to/audio.wav \
        --output-dir output/segments/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import librosa
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_whisper_segments(whisper_json_path: Path) -> list[dict[str, Any]]:
    """Whisperの文字起こし結果JSONからセグメント情報を読み込む

    Args:
        whisper_json_path: Whisperの出力JSONファイルのパス

    Returns:
        セグメント情報のリスト

    Raises:
        FileNotFoundError: JSONファイルが見つからない場合
        json.JSONDecodeError: JSONの解析に失敗した場合
    """
    if not whisper_json_path.exists():
        raise FileNotFoundError(f"Whisper JSON file not found: {whisper_json_path}")

    with whisper_json_path.open(encoding="utf-8") as f:
        segments = json.load(f)

    logger.info(f"Loaded {len(segments)} segments from {whisper_json_path}")
    return segments


def sanitize_filename(text: str) -> str:
    """ファイル名として使用できるように文字列をサニタイズする

    Args:
        text: 元のテキスト

    Returns:
        サニタイズされたファイル名
    """
    # 不正文字を除去し、長すぎる場合は切り詰める
    sanitized = "".join(c for c in text if c.isalnum() or c in (" ", "-", "_")).strip()
    # 連続するスペースを単一のアンダースコアに置換
    sanitized = "_".join(sanitized.split())
    # 最大50文字に制限
    if len(sanitized) > 50:
        sanitized = sanitized[:50]
    return sanitized or "segment"


def split_audio_by_segments(
    wav_file: Path, segments: list[dict[str, Any]], output_dir: Path
) -> None:
    """音声ファイルをセグメントに基づいて分割する

    Args:
        wav_file: 入力音声ファイルのパス
        segments: Whisperセグメント情報のリスト
        output_dir: 出力ディレクトリのパス

    Raises:
        FileNotFoundError: 音声ファイルが見つからない場合
        Exception: 音声処理に失敗した場合
    """
    if not wav_file.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    # 音声ファイルを読み込み（librosaで自動的にモノラル変換）
    try:
        audio_data, sample_rate = librosa.load(str(wav_file), sr=None, mono=True)
        duration_seconds = len(audio_data) / sample_rate
        logger.info(
            f"Loaded audio file: {wav_file} "
            f"(duration: {duration_seconds:.2f}s, sr: {sample_rate}Hz, "
            f"shape: {audio_data.shape})"
        )
    except Exception as e:
        raise Exception(f"Failed to load audio file: {e}") from e

    for i, segment in enumerate(segments):
        start_time = segment["start"]
        original_end_time = segment["end"]
        text = segment["text"].strip()

        # 次のセグメントの開始時間を取得（最後のセグメントの場合はNone）
        next_start_time = segments[i + 1]["start"] if i + 1 < len(segments) else None

        # 終了時間を調整: 元の終了時間+1秒と次のセグメントの開始時間の小さい方
        adjusted_end_time = original_end_time + 1.0
        if next_start_time is not None:
            adjusted_end_time = min(adjusted_end_time, next_start_time)

        # ファイル名を生成
        sanitized_text = sanitize_filename(text)
        output_filename = (
            f"{i + 1:03d}_{start_time:.2f}-{adjusted_end_time:.2f}_{sanitized_text}.wav"
        )
        output_path = output_dir / output_filename

        try:
            # 時間をサンプル数に変換
            start_sample = int(start_time * sample_rate)
            end_sample = int(adjusted_end_time * sample_rate)
            
            # 音声の範囲をチェック
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)

            if start_sample >= end_sample:
                logger.warning(f"Skipping segment {i + 1}: invalid time range")
                continue

            # 音声セグメントを抽出
            segment_audio = audio_data[start_sample:end_sample]

            # WAVファイルとして保存
            sf.write(str(output_path), segment_audio, sample_rate, subtype='PCM_16')

            logger.info(
                f"Created segment {i + 1}/{len(segments)}: {output_filename} "
                f"(original: {original_end_time:.2f}s "
                f"→ adjusted: {adjusted_end_time:.2f}s, "
                f"samples: {len(segment_audio)})"
            )

        except Exception as e:
            logger.error(f"Failed to create segment {i + 1}: {e}")
            continue

    logger.info(f"Audio splitting completed. Output files saved to: {output_dir}")


def main() -> int:
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="音声ファイルをWhisperの文字起こし結果のセグメントに基づいて分割する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    uv run scripts/split_audio_by_segments.py \\
        --whisper-json local/whisper_transcription.json \\
        --wav-file path/to/audio.wav \\
        --output-dir output/segments/
        """,
    )

    parser.add_argument(
        "--whisper-json",
        type=Path,
        required=True,
        help="Whisperの文字起こし結果JSONファイルのパス",
    )

    parser.add_argument(
        "--wav-file",
        type=Path,
        required=True,
        help="分割する音声ファイル（WAV形式）のパス",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="分割された音声ファイルの出力ディレクトリのパス",
    )

    args = parser.parse_args()

    # Whisperセグメントを読み込み
    segments = load_whisper_segments(args.whisper_json)

    # 音声を分割
    split_audio_by_segments(args.wav_file, segments, args.output_dir)

if __name__ == "__main__":
    main()