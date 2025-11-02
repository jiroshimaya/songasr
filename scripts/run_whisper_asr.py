#!/usr/bin/env python3
"""Whisper ASRスクリプト - 音声ファイルを文字起こしします"""

import argparse
import json
import sys
from pathlib import Path

from faster_whisper import WhisperModel


def main():
    parser = argparse.ArgumentParser(
        description="音声ファイルをWhisperで文字起こしします"
    )
    parser.add_argument("input_path", help="入力音声ファイルのパス")
    parser.add_argument("output_path", help="出力JSONファイルのパス")
    parser.add_argument(
        "--model", default="medium", help="Whisperモデル名 (default: medium)"
    )
    parser.add_argument(
        "--device", default="cpu", help="デバイス (cpu/cuda) (default: cpu)"
    )

    args = parser.parse_args()

    # 入力ファイルの存在確認
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_path}", file=sys.stderr)
        sys.exit(1)

    # 出力ディレクトリの作成
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"音声ファイル: {input_path}")
    print(f"出力ファイル: {output_path}")
    print(f"モデル: {args.model}")
    print(f"デバイス: {args.device}")

    model = WhisperModel(args.model, device=args.device)
    segments, _ = model.transcribe(str(input_path), word_timestamps=True)

    # セグメント（開始/終了・テキスト）
    with output_path.open("w", encoding="utf-8") as f:
        json_segments = []
        for s in segments:
            json_s = {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end} for w in s.words
                ]
                if s.words
                else [],
            }
            json_segments.append(json_s)
        json.dump(json_segments, f, ensure_ascii=False, indent=2)

    for s in segments:
        print(f"{s.start:.2f} --> {s.end:.2f} : {s.text}")
        # 単語ごとのタイムスタンプ
        if s.words:
            for w in s.words:
                print(f"  [{w.start:.2f}-{w.end:.2f}] {w.word}")


if __name__ == "__main__":
    main()
