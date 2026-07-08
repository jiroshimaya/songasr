#!/usr/bin/env python3
"""楽曲ファイル → 歌える空耳カナ歌詞シート (issue #7 の成果の実用CLI)。

パイプライン: faster-whisper large-v3 全曲転写 (ASR自身のセグメント単位)
             → 単語ごとに e2k でカナ化 → 空耳縮約ルール適用

使い方 (GPU機のリポジトリルートで、LD_LIBRARY_PATH設定済みの状態で):
  PYTHONPATH=src:scripts/exp7 .venv/bin/python scripts/exp7/song_to_kana.py \
      <audio.(wav|mp3|m4a)> [-o out.txt] [--no-english] [--device cuda]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from soramimi_contract import contract_words


def fmt_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--no-english", action="store_true", help="英語行を出さない")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--model", default="large-v3")
    args = ap.parse_args()

    import re

    from e2k import C2K
    from faster_whisper import WhisperModel

    c2k = C2K()

    def to_kana_line(text: str) -> str:
        words = [w for w in re.findall(r"[a-zA-Z']+", text.lower()) if w.strip("'")]
        kanas = []
        for w in words:
            try:
                kanas.append(c2k(w.replace("'", "")))
            except Exception:  # noqa: BLE001
                kanas.append("")
        return " ".join(k for k in contract_words(words, kanas) if k)

    compute = "float16" if args.device == "cuda" else "int8"
    model = WhisperModel(args.model, device=args.device, compute_type=compute)
    segments, info = model.transcribe(
        str(args.audio), language="en", vad_filter=False, beam_size=5)

    lines = []
    for seg in segments:
        kana = to_kana_line(seg.text.strip())
        if not kana:
            continue
        header = f"[{fmt_ts(seg.start)}-{fmt_ts(seg.end)}]"
        if args.no_english:
            lines.append(f"{header} {kana}")
        else:
            lines.append(f"{header} {kana}\n{'':>13s}({seg.text.strip()})")
        print(lines[-1], flush=True)

    if args.out:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
