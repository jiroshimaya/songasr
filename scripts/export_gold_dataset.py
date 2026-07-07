#!/usr/bin/env python3
"""align_song_kana.py の出力(区間+カナ+予測)を、自己完結した「正解(gold)データセット」に整形する。

- 高信頼マッチ(score>=--min-score)の区間だけを採用
- 各区間を 16kHz mono wav クリップに切り出し clips/ へ保存
- manifest.jsonl (1行1クリップ, スキーマは README 参照) を書き出す
- README.md に、スキーマ・生成手順・出所/ライセンス上の注意・現行結果・別セッションへのタスク説明を書く

これにより「音声クリップ + 正解カナ + 現行モデル予測/誤差」が1ディレクトリに揃い、
別の新規セッションに渡してゼロベースで精度改善方法を検討してもらえる。

使い方:
  uv run python scripts/export_gold_dataset.py \
      --aligned local/yougaku/stand_by_me_aligned.json \
      --audio   local/yougaku/stand_by_me_16k.wav \
      --song    "Stand By Me" \
      --recording "John Lennon - Rock 'N' Roll" \
      --outdir  local/gold_dataset
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import soundfile as sf

TARGET_SR = 16_000


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--song", required=True)
    ap.add_argument("--recording", required=True, help="購入した実際の録音(版)")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--min-score", type=float, default=0.6)
    args = ap.parse_args()

    recs = json.loads(args.aligned.read_text(encoding="utf-8"))
    audio, sr = sf.read(str(args.audio), dtype="float32", always_2d=False)
    assert sr == TARGET_SR, f"想定は16kHz, 実際は{sr}"

    clips_dir = args.outdir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    song_slug = slugify(args.song)

    manifest = []
    kept = 0
    for r in recs:
        if r["match_score"] < args.min_score:
            continue
        kept += 1
        cid = f"{song_slug}_{kept:04d}"
        s = max(0, int(r["start"] * TARGET_SR))
        e = min(len(audio), int(r["end"] * TARGET_SR))
        clip_path = clips_dir / f"{cid}.wav"
        sf.write(str(clip_path), audio[s:e], TARGET_SR, subtype="PCM_16")
        manifest.append({
            "id": cid,
            "audio_file": f"clips/{cid}.wav",
            "song": args.song,
            "recording": args.recording,
            "start_sec": round(r["start"], 3),
            "end_sec": round(r["end"], 3),
            "duration_sec": round((e - s) / TARGET_SR, 3),
            "english_sung": r["whisper_en"],            # whisperが聞き取った歌唱の英語
            "english_ref": r["matched_yougaku_en"],     # yougakuの対応英語行
            "kana": r["kana_ref"],                       # ★正解カナ(yougaku existKana)
            "kana_source": "yougaku-nihongo.com",
            "match_score": r["match_score"],
            "baseline_pred": r.get("baseline_kana"),
            "baseline_kanasim": r.get("baseline_dist"),
            "finetuned_pred": r.get("finetuned_kana"),
            "finetuned_kanasim": r.get("finetuned_dist"),
        })

    man_path = args.outdir / "manifest.jsonl"
    with man_path.open("w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # 現行結果の集計
    def mean(key: str) -> float | None:
        vs = [m[key] for m in manifest if m.get(key) is not None]
        return round(sum(vs) / len(vs), 1) if vs else None

    n = len(manifest)
    b_n = sum(1 for m in manifest if m["baseline_kanasim"] is not None)
    f_n = sum(1 for m in manifest if m["finetuned_kanasim"] is not None)
    readme = _README.format(
        song=args.song, recording=args.recording, n=n,
        b_mean=mean("baseline_kanasim"), b_n=b_n,
        f_mean=mean("finetuned_kanasim"), f_n=f_n,
        min_score=args.min_score,
    )
    (args.outdir / "README.md").write_text(readme, encoding="utf-8")

    print(f"gold clips: {n} 件 -> {clips_dir}")
    print(f"manifest: {man_path}")
    print(f"baseline 採点可 {b_n}/{n} 平均kanasim={mean('baseline_kanasim')}")
    print(f"finetuned 採点可 {f_n}/{n} 平均kanasim={mean('finetuned_kanasim')}")


_README = """# 正解データセット: 英語歌唱 → 日本語カナ (soramimi ASR)

英語楽曲の歌唱音声を、日本語カタカナ発音（歌える"空耳"表記）に書き起こすASRのための
**正解(gold)データセット**。1区間 = (音声クリップ, 正解カナ) のペア。

## 収録
- 曲: **{song}**
- 録音(版): **{recording}**（正規購入した音源）
- クリップ数: **{n}** 件（16kHz mono wav, `clips/`）
- マッチ信頼度しきい値: score>={min_score}

## スキーマ (manifest.jsonl, 1行1クリップ)
| フィールド | 意味 |
|---|---|
| id | クリップID |
| audio_file | クリップwavへの相対パス(16kHz mono) |
| song / recording | 曲名 / 購入した録音の版 |
| start_sec / end_sec / duration_sec | 元音源中の区間 |
| english_sung | whisperが聞き取った歌唱の英語 |
| english_ref | 対応するyougakuの英語歌詞行 |
| **kana** | **★正解カナ**(yougaku由来, 人手作成の歌えるカタカナ) |
| kana_source | 正解カナの出所 |
| match_score | english_sung と english_ref のファジー一致度 |
| baseline_pred / baseline_kanasim | baseline疑似ラベラーの予測 / 正解との kanasim 距離(小さいほど良, Noneは採点不能) |
| finetuned_pred / finetuned_kanasim | #3で学習したLoRAモデルの予測 / kanasim距離 |

## 生成手順
1. 購入音源を16kHz mono wavへ変換
2. faster-whisper(large-v3, word timestamps, VADなし)で歌唱を英語書き起こし→区間化
3. 各区間の英語をyougakuの英語歌詞行にファジーマッチし、対応する**正解カナ**を付与
4. 各区間を切り出し、baseline / 学習済みモデルで認識して kanasim 距離を付記
(スクリプト: scripts/align_song_kana.py, scripts/export_gold_dataset.py)

## 現行の精度(このデータ上)
- baseline: 平均 kanasim {b_mean} (採点可 {b_n}/{n})
- finetuned(#3): 平均 kanasim {f_mean} (採点可 {f_n}/{n})
- kanasim が None なのは、予測が壊れて距離表に載らないモーラ列になったケース(=実質失敗)。

## 既知の限界
- フルバンド伴奏の干渉で認識が大きく崩れる（採点不能が多い）。ボーカル分離(Demucs)は要検証。
- サビの繰り返し行が重複して入る（同一カナが複数区間に対応）。
- 現状1曲のみ・話者/ジャンルの多様性なし。

## 出所・ライセンス上の注意（重要）
- **音声**: 正規購入した楽曲を、日本の著作権法30条の4（情報解析）に基づき**研究目的の解析にのみ**使用。**再配布不可**。
- **正解カナ**: yougaku-nihongo.com の編集コンテンツ由来。**研究利用に留め再配布しない**前提。公開・商用は別途権利確認が必要。
- したがって本データセットは**外部公開・配布しない**こと。

## 別セッションへのタスク（ゼロベース検討用）
「英語歌唱→日本語カナ」の認識精度を上げる方法を、この正解データを使って検討してほしい。
現状: baseline(IPA→ARPABET→カナ)と、TTS疑似ラベルで学習した1B wav2vec2 CTCのLoRA(#3)がある。
実歌唱ではフルバンド伴奏で認識が崩れるのが最大の課題。manifestの baseline_pred/finetuned_pred と
正解 kana を突き合わせ、誤りの傾向分析と改善アプローチ（前処理/データ/モデル/デコード等）を提案してほしい。
"""


if __name__ == "__main__":
    main()
