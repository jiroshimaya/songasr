#!/usr/bin/env python3
"""
Whisperの単語単位のタイムスタンプに基づいて音声を分割するスクリプト

使用例:
    uv run scripts/split_audio_by_words.py \
        --whisper-json local/whisper_transcription.json \
        --wav-file path/to/audio.wav \
        --output-dir output/words/ \
        [--padding 0.1]
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


def load_whisper_words(whisper_json_path: Path) -> list[dict[str, Any]]:
    """WhisperのJSONから単語情報を抽出する"""
    segments = load_whisper_segments(whisper_json_path)
    words: list[dict[str, Any]] = []

    for segment_index, segment in enumerate(segments):
        segment_words = segment.get("words", [])
        if not segment_words:
            logger.warning("Segment %d has no word-level timestamps", segment_index + 1)
            continue

        for word_info in segment_words:
            start = word_info.get("start")
            end = word_info.get("end")
            text = word_info.get("word", "").strip()

            if start is None or end is None:
                logger.warning(
                    "Skipping word without timestamps in segment %d: %s",
                    segment_index + 1,
                    word_info,
                )
                continue

            start_f = float(start)
            end_f = float(end)
            if start_f >= end_f:
                logger.warning(
                    "Skipping word with invalid time range in segment %d: %s",
                    segment_index + 1,
                    word_info,
                )
                continue

            words.append(
                {
                    "start": start_f,
                    "end": end_f,
                    "text": text,
                    "segment_index": segment_index,
                }
            )

    if not words:
        raise ValueError("No valid word-level timestamps found in Whisper JSON")

    logger.info("Extracted %d words from %s", len(words), whisper_json_path)
    return words


def split_audio_by_words(
    wav_file: Path,
    words: list[dict[str, Any]],
    output_dir: Path,
    padding: float,
) -> None:
    """音声ファイルを単語単位で分割する"""
    if not wav_file.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_file}")

    safe_padding = max(0.0, padding)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created output directory: %s", output_dir)

    try:
        audio_data, sample_rate = librosa.load(str(wav_file), sr=None, mono=True)
        duration_seconds = len(audio_data) / sample_rate
        logger.info(
            "Loaded audio file: %s (duration: %.2fs, sr: %dHz, shape: %s)",
            wav_file,
            duration_seconds,
            sample_rate,
            audio_data.shape,
        )
    except Exception as exc:
        raise Exception(f"Failed to load audio file: {exc}") from exc

    for i, word in enumerate(words):
        base_start = float(word["start"])
        base_end = float(word["end"])
        next_start_time = float(words[i + 1]["start"]) if i + 1 < len(words) else None
        adjusted_end_time = base_end + 1.0
        if next_start_time is not None:
            adjusted_end_time = min(adjusted_end_time, next_start_time)

        start_time = max(0.0, base_start - safe_padding)
        padded_end_time = min(duration_seconds, adjusted_end_time + safe_padding)
        end_time = (
            min(padded_end_time, next_start_time)
            if next_start_time is not None
            else padded_end_time
        )

        text = word.get("text", "").strip()

        if start_time >= end_time:
            logger.warning("Skipping word %d: invalid time range", i + 1)
            continue

        segment_index = word["segment_index"] + 1
        sanitized_text_raw = sanitize_filename(text)
        sanitized_text = sanitized_text_raw or f"word_{i + 1}"
        time_range = f"{start_time:.2f}-{end_time:.2f}"
        output_filename = (
            f"{i + 1:04d}_seg{segment_index:03d}_{time_range}_{sanitized_text}.wav"
        )
        output_path = output_dir / output_filename

        try:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)

            if start_sample >= end_sample:
                logger.warning("Skipping word %d: invalid sample range", i + 1)
                continue

            word_audio = audio_data[start_sample:end_sample]
            sf.write(str(output_path), word_audio, sample_rate, subtype="PCM_16")

            logger.info(
                "Created word %d/%d (segment %d): %s "
                "(start: %.2fs, final end: %.2fs, adjusted end: %.2fs, samples: %d)",
                i + 1,
                len(words),
                segment_index,
                output_filename,
                start_time,
                end_time,
                adjusted_end_time,
                len(word_audio),
            )
        except Exception as exc:
            logger.error("Failed to create word %d: %s", i + 1, exc)
            continue

    logger.info(
        "Word-level audio splitting completed. Output files saved to: %s", output_dir
    )


def main() -> None:
    """CLIエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="Whisperの単語タイムスタンプに基づいて音声ファイルを分割する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    uv run scripts/split_audio_by_words.py \\
        --whisper-json local/whisper_transcription.json \\
        --wav-file path/to/audio.wav \\
        --output-dir output/words/ \\
        [--padding 0.1]
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

    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="各単語の前後に付与する余白秒数 (デフォルト: 0.0)",
    )

    args = parser.parse_args()

    words = load_whisper_words(args.whisper_json)
    split_audio_by_words(args.wav_file, words, args.output_dir, args.padding)


if __name__ == "__main__":
    main()
