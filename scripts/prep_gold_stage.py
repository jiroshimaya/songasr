#!/usr/bin/env python3
"""Mac側: 購入したアルバムから yougakuカナ有りのトラックだけを抽出し、
16kHz mono wav へ変換 + yougakuレコード(json)を添えて staging ディレクトリに揃える。

- アルバムフォルダ内の .m4a を走査し、曲名を正規化して yougaku(784曲)にカナがあるものだけ採用
- afconvert で 16kHz mono wav に変換 (macOS標準, ffmpeg不要)
- 各曲の yougakuレコードを <slug>.json に保存
- stage.jsonl に {slug, wav, yougaku, song, recording} を書き出す
(この後 rsync で GPU機へ送り、build_gold_batch.py で一括処理する)
"""

from __future__ import annotations

import json
import re
import subprocess
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
YOUGAKU = REPO / "local/yougaku/songsLyrics3.json"
STAGE = REPO / "local/gold_stage"

# (アルバムフォルダ, recordingラベル)
ALBUMS = [
    (
        Path.home() / "Music/Music/Media.localized/エド・シーラン/÷ (Deluxe)",
        "Ed Sheeran - Divide (Deluxe)",
    ),
    (
        Path.home() / "Music/Music/Media.localized/Pasek & Paul, ヒュー・ジャックマン, "
        "ザック・エフロン & ゼンデイヤ/グレイテスト・ショーマン(オリジナル・サウンドトラック)",
        "The Greatest Showman OST",
    ),
]

# 既に用意済みの単曲 (wav, yougaku json, recording)
EXTRA = [
    (
        REPO / "local/yougaku/stand_by_me_16k.wav",
        REPO / "local/yougaku/stand_by_me.json",
        "Stand By Me",
        "John Lennon - Rock N Roll",
    ),
]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[（(].*?[)）]", "", s)
    s = s.lower().replace("’", "").replace("'", "").replace("&", "and")
    s = re.sub(r"ft\.?.*$", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def track_title(m4a: Path) -> str:
    """'01 Galway Girl.m4a' -> 'Galway Girl'"""
    stem = m4a.stem
    return re.sub(r"^\d+\s+", "", stem).strip()


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm(s)).strip("_")


def main() -> None:
    records = json.loads(YOUGAKU.read_text(encoding="utf-8"))
    by_norm = {norm(d["title"]): d for d in records if str(d.get("kana", "")).strip()}

    STAGE.mkdir(parents=True, exist_ok=True)
    stage = []
    seen_slugs = set()

    for album_dir, recording in ALBUMS:
        if not album_dir.exists():
            print(f"!! アルバムフォルダなし: {album_dir}")
            continue
        for m4a in sorted(album_dir.glob("*.m4a")):
            title = track_title(m4a)
            rec = by_norm.get(norm(title))
            if rec is None:
                continue  # yougakuにカナが無い曲はスキップ
            slug = slugify(title)
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            wav = STAGE / f"{slug}_16k.wav"
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(m4a), str(wav)],
                check=True,
            )
            (STAGE / f"{slug}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            stage.append({
                "slug": slug, "wav": wav.name, "yougaku": f"{slug}.json",
                "song": rec["title"], "recording": recording,
            })
            print(f"  + {title}  ->  {slug}")

    # 既存単曲を取り込み
    for wav_src, json_src, song, recording in EXTRA:
        if not wav_src.exists():
            print(f"!! 単曲wavなし: {wav_src}")
            continue
        slug = slugify(song)
        wav = STAGE / f"{slug}_16k.wav"
        if wav_src.resolve() != wav.resolve():
            wav.write_bytes(wav_src.read_bytes())
        (STAGE / f"{slug}.json").write_text(json_src.read_text(encoding="utf-8"), encoding="utf-8")
        stage.append({
            "slug": slug, "wav": wav.name, "yougaku": f"{slug}.json",
            "song": song, "recording": recording,
        })
        print(f"  + {song} (extra)  ->  {slug}")

    (STAGE / "stage.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in stage) + "\n", encoding="utf-8"
    )
    print(f"\nステージ完了: {len(stage)} 曲 -> {STAGE}")


if __name__ == "__main__":
    main()
