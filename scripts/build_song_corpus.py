#!/usr/bin/env python3
"""Pixabayの商用利用可トラック(既存の1曲)から、全27フレーズ区間を切り出し、
baseline / wav2vec2-ja の両方を実行してkanasimでリファレンスと比較する。

新規トラックの発掘(pixabay.comの検索・閲覧)はCloudflareのbot対策で
自動化ツールからは全面的にブロックされていた(docs/kana-asr-experiment/
DATASET_NOTES.md 参照)ため、既存の1曲内でセグメント数を増やす方針に切り替えている。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf
from kanasim import create_kana_distance_calculator

from songasr.kana_baseline import baseline_transcribe
from songasr.kana_wav2vec2_ja import wav2vec2_ja_transcribe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_SR = 16_000
SONG_MP3 = Path("local/song.mp3")
SONG_URL = "https://cdn.pixabay.com/download/audio/2025/03/22/audio_f46c5fa5ad.mp3"
OUT_DIR = Path("local/song_corpus")
PAD_BEFORE, PAD_AFTER = 0.2, 0.3

# openai-whisper (English, word_timestamps=True) で得た全27区間の書き起こしと、
# 標準的な外来語カナ表記によるリファレンス(手書き)。
SEGMENTS = [
    (4.30, 11.12, "Under the moon and the quiet sky so clear", "アンダーザムーンアンドザクワイエットスカイソークリア"),
    (12.00, 17.66, "A whisper of hope is all I hold near", "アウィスパーオブホープイズオールアイホールドニア"),
    (17.66, 23.50, "Every breath, every word", "エヴリブレスエヴリワード"),
    (23.50, 30.20, "A humble call, guided by mercy", "アハンブルコールガイデッドバイマーシー"),
    (30.20, 33.28, "The one above all", "ザワンアバヴオール"),
    (37.28, 42.76, "Through the path with thorns where shadows creep", "スルーザパスウィズソーンズウェアシャドウズクリープ"),
    (43.50, 49.72, "I tread with faith, my soul does leave", "アイトレッドウィズフェイスマイソウルダズリーヴ"),
    (50.72, 55.64, "Hands raised high with the hearts in tears", "ハンズレイズドハイウィズザハーツインティアーズ"),
    (56.72, 63.00, "Every moment, I know he's near", "エヴリモーメントアイノウヒーズニア"),
    (63.72, 69.72, "Oh eternal light, that shines so pure", "オーエターナルライトザットシャインズソーピュア"),
    (69.72, 75.30, "Guide me always, your love is sure", "ガイドミーオールウェイズユアラヴイズシュア"),
    (76.30, 82.48, "No reaches, I see, no crowd to wear", "ノーリーチズアイシーノークラウドトゥウェア"),
    (82.48, 85.84, "Just your mercy forever", "ジャストユアマーシーフォーエヴァー"),
    (87.56, 93.74, "Everywhere in the stillness where the silence stays", "エヴリウェアインザスティルネスウェアザサイレンスステイズ"),
    (94.30, 100.24, "I find my solace in whispered praise", "アイファインドマイソラスインウィスパードプレイズ"),
    (101.78, 106.66, "From the sunrise to the starlit dawn", "フロムザサンライズトゥザスターリットドーン"),
    (107.24, 113.82, "In your grace, I found my home", "インユアグレイスアイファウンドマイホーム"),
    (113.82, 119.88, "When my voice shakes and my fears ignite", "ウェンマイヴォイスシェイクスアンドマイフィアーズイグナイト"),
    (119.88, 126.12, "You calm the storm with the warmth, your light", "ユーカームザストームウィズザウォームスユアライト"),
    (127.12, 133.76, "In the echoes of faith, I find my song", "インジエコーズオブフェイスアイファインドマイソング"),
    (133.76, 138.92, "With you, I'll stand forever strong", "ウィズユーアイルスタンドフォーエヴァーストロング"),
    (140.12, 146.56, "Oh eternal light, that shines so pure", "オーエターナルライトザットシャインズソーピュア"),
    (147.12, 151.58, "Guide me always, your love is sure", "ガイドミーオールウェイズユアラヴイズシュア"),
    (152.58, 159.28, "No reaches, I see, no crowd to wear", "ノーリーチズアイシーノークラウドトゥウェア"),
    (159.28, 162.80, "Just your mercy forever", "ジャストユアマーシーフォーエヴァー"),
    (163.58, 165.40, "Everywhere", "エヴリウェア"),
    (181.70, 183.10, "Oh", "オー"),
]


def ensure_song() -> np.ndarray:
    if not SONG_MP3.exists():
        SONG_MP3.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(SONG_URL, timeout=30)
        resp.raise_for_status()
        SONG_MP3.write_bytes(resp.content)
    audio, sr = sf.read(str(SONG_MP3), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32)


def extract(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    s = max(0, int((start - PAD_BEFORE) * TARGET_SR))
    e = min(len(audio), int((end + PAD_AFTER) * TARGET_SR))
    return audio[s:e]


def safe_distance(calc, kana: str, ref: str) -> float | None:
    """kanasimは一部のモーラ組み合わせ(不自然な連続など)を距離表に持たず
    KeyErrorになることがあるため、その場合はNone(採点不能)として扱う。"""
    if not kana:
        return None
    try:
        return calc.calculate(kana, ref)
    except KeyError as e:
        logger.warning("kanasim distance failed for %r vs %r: %s", kana, ref, e)
        return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    song_audio = ensure_song()
    calc = create_kana_distance_calculator()

    records = []
    for i, (start, end, text, ref_kana) in enumerate(SEGMENTS, 1):
        clip = extract(song_audio, start, end)
        base = baseline_transcribe(clip, TARGET_SR)
        w2v = wav2vec2_ja_transcribe(clip, TARGET_SR)
        base_dist = safe_distance(calc, base["kana"], ref_kana)
        w2v_dist = safe_distance(calc, w2v["kana"], ref_kana)

        logger.info(
            "[%02d/%d] %s | base=%s(%s) w2v=%s(%s) ref=%s",
            i, len(SEGMENTS), text, base["kana"], base_dist, w2v["kana"], w2v_dist, ref_kana,
        )
        records.append({
            "index": i, "start": start, "end": end, "text": text,
            "reference_kana": ref_kana,
            "baseline_kana": base["kana"], "baseline_distance": base_dist,
            "wav2vec2_ja_kana": w2v["kana"], "wav2vec2_ja_distance": w2v_dist,
        })

    Path(OUT_DIR / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))

    valid_base = [r["baseline_distance"] for r in records if r["baseline_distance"] is not None]
    valid_w2v = [r["wav2vec2_ja_distance"] for r in records if r["wav2vec2_ja_distance"] is not None]
    print(f"\nn={len(records)}")
    print(f"baseline: mean={sum(valid_base)/len(valid_base):.2f} n_valid={len(valid_base)}")
    print(f"wav2vec2-ja: mean={sum(valid_w2v)/len(valid_w2v):.2f} n_valid={len(valid_w2v)}")
    print(f"baseline wins (lower dist): {sum(1 for r in records if r['baseline_distance'] is not None and r['wav2vec2_ja_distance'] is not None and r['baseline_distance'] < r['wav2vec2_ja_distance'])} / {len(records)}")


if __name__ == "__main__":
    main()
