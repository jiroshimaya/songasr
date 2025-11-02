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
import sys
from pathlib import Path
from typing import Any

from pydub import AudioSegment

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

    # 音声ファイルを読み込み
    try:
        audio = AudioSegment.from_file(str(wav_file))
        logger.info(
            f"Loaded audio file: {wav_file} (duration: {len(audio) / 1000:.2f}s)"
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
            # 時間をミリ秒に変換
            start_ms = int(start_time * 1000)
            end_ms = int(adjusted_end_time * 1000)

            # 音声セグメントを抽出
            segment_audio = audio[start_ms:end_ms]

            # WAVファイルとして保存
            segment_audio.export(str(output_path), format="wav")

            logger.info(
                f"Created segment {i + 1}/{len(segments)}: {output_filename} "
                f"(original: {original_end_time:.2f}s "
                f"→ adjusted: {adjusted_end_time:.2f}s)"
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

    try:
        # Whisperセグメントを読み込み
        segments = load_whisper_segments(args.whisper_json)

        # 音声を分割
        split_audio_by_segments(args.wav_file, segments, args.output_dir)

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
