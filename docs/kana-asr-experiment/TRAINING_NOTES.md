# モデル学習の記録 (issue #3)

[トラッキングissue #5](https://github.com/jiroshimaya/songasr/issues/5) の3ステップのうち [#3 モデル学習](https://github.com/jiroshimaya/songasr/issues/3) に着手した記録。[#2 正解データ作成](./DATASET_NOTES.md) で用意した TTS疑似ラベル(v2, 1000文) を教師データに、英語歌唱→カナ用のCTCモデルをLoRAでファインチューニングし、実歌唱27区間ホールドアウトで既存手法(baseline / wav2vec2-ja直接推論)と比較した。

**結論(先に要約)**: TTS疑似ラベルのみで学習したモデルが、学習に使っていない実歌唱ホールドアウトで既存2手法を平均kanasim距離で上回った。3手法とも採点可能な共通20区間で、finetuned=86.69 < baseline=93.81 < wav2vec2-ja=107.20。勝敗は finetuned が baseline に 12/20、直接推論に 17/20。弱教師ラベル・小標本という限界の中での結果である。

## 実行環境

- GPU機 `jiro-FRONTIER` (RTX 4060 Ti, 16GB)、torch 2.9.0+cu128。
- 依存は `uv sync` + 学習用に `peft` / `accelerate` / `jiwer` を追加。
- 学習中のGPUメモリ使用は約7GB / 16GB(バッチ4, bf16)で、16GBに十分収まった。

## 手法: LoRAファインチューニング

1Bパラメータの `AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana` を16GBで学習するため、フルFTではなくLoRAを採用した(`scripts/train_kana_ctc.py`)。弱教師ラベルに対しフルFTだとbaselineの癖ごと過学習しやすいため、パラメータを絞るLoRAは正則化の観点でも妥当と判断。

- **凍結/学習**: feature encoder は凍結(`freeze_feature_encoder`)。LoRA を注意層の `q_proj,k_proj,v_proj,out_proj` に適用(r=16, alpha=32, dropout=0.05)、CTCヘッド `lm_head` は `modules_to_save` で学習対象に残す。
- **学習可能パラメータ**: 8,096,181 / 970,825,450 (**0.83%**)。
- **省メモリ**: bf16 autocast、gradient checkpointing(`use_reentrant=False`)。※wav2vec2は入力埋め込みを持たないため `enable_input_require_grads` は使えず、非reentrantチェックポイントで代替。
- **語彙**: ラベルの仮名75種はベースモデルの語彙(181トークン)に全て含まれOOV=0。トークナイザ拡張・ヘッドのリサイズは不要だった。

### データ

- 教師: `data/kana_dataset_v2/manifest.jsonl`(1000文)。音声は `scripts/build_kana_dataset_v2.py --n 1000` で `local/kana_dataset_v2/audio/` に再生成。train 950 / val 50(seed=42)。
- ラベルは baseline 疑似ラベラー由来の**弱教師あり**であり、質はbaselineの精度に依存する([DATASET_NOTES.md](./DATASET_NOTES.md) の限界がそのまま残る)。
- **再生成の非決定性(注記)**: gTTS(ネットワークTTS)の合成音声は実行ごとに微妙に変わり、境界ケースのbaseline疑似ラベルが変化する。今回の再生成では committed 版manifestと **61/1000行(約6%)** が相違した。音声とラベルは同一runで整合しており学習上は問題ないが、committed manifestとの完全なbit一致は保証されない。

### ハイパーパラメータ

| 項目 | 値 |
|---|---|
| epochs | 5 |
| batch (per device) | 4 |
| gradient accumulation | 2 (実効バッチ8) |
| learning rate | 3e-4 |
| warmup ratio | 0.1 |
| precision | bf16 |

学習時間は約 **380秒 (6.3分)**、595ステップ。

## 学習経過 (弱教師valに対するCER)

| epoch | eval_loss | eval_cer |
|---|---|---|
| 1 | 0.986 | 0.474 |
| 2 | 0.674 | 0.387 |
| 3 | 0.523 | 0.354 |
| 4 | 0.480 | 0.333 |
| 5 | 0.452 | **0.327** |

CERは単調に低下。ただしこれは**疑似ラベルに対する**一致度であり、真の正しさではなく「baselineのラベルにどれだけ寄ったか」を測っている点に注意。

## ホールドアウト評価 (実歌唱27区間, kanasim距離; 小さいほど良い)

学習に一切使っていない実歌唱1曲・27区間([`song_corpus_results.json`](./song_corpus_results.json) と同一区間・同一リファレンス)で評価。

### 各手法の全区間平均(採点可能な区間のみ、母数は手法ごとに異なる)

| 手法 | 平均距離 | n |
|---|---|---|
| wav2vec2-ja 直接推論 | 105.13 | 24 |
| baseline 疑似ラベラー | 88.15 | 25 |
| **finetuned (本手法)** | **83.10** | 23 |

kanasimは一部のモーラ列で距離表にKeyErrorを起こし採点不能になるため、手法ごとに有効区間数(n)が異なる。上の平均は母数が揃っておらず厳密な比較にはならない。

### 3手法とも採点可能な共通20区間でのペア比較(母数を揃えた公平な比較)

| 手法 | 共通20区間の平均距離 |
|---|---|
| wav2vec2-ja 直接推論 | 107.20 |
| baseline 疑似ラベラー | 93.81 |
| **finetuned (本手法)** | **86.69** |

- **finetuned < baseline: 12/20 (60%)**
- **finetuned < wav2vec2-ja: 17/20 (85%)**

## 考察と限界

- **効果は本物だが控えめ**: TTS疑似ラベルのみで学習したモデルが、ドメインの異なる実歌唱で両ベースラインを平均で上回ったのは、#3のアプローチ(専用モデルの学習)が機能する実証。特に元モデルの直接推論に対しては 85% の区間で改善。
- **baseline比は中程度**: 教師がそのbaselineの疑似ラベルなので、baselineを大きく超えるのは原理的に難しい(弱教師の天井)。それでも平均で下回り、60%の区間で勝ったのは、複数アクセント(tld5種)・1000文への汎化で個々のノイズが均された効果と解釈できる。
- **小標本**: 共通20区間・1曲・1歌手のみ。統計的な強い主張はできず、傾向を見る段階。
- **弱教師の限界は未解決**: 真の正解ではなくbaselineラベルで学習しているため、baseline自体の系統誤差は引き継ぐ。

## 今後の候補

1. 実歌唱データの拡充(検証だけでなく学習にも使えるように; 商用利用可ソースの確保が課題)。
2. 歌唱に忠実なデータ拡張(母音だけ伸長・ビブラート等; [DATASET_NOTES.md](./DATASET_NOTES.md) の一様pitch/tempoでは不十分)。
3. LoRA rank / target module / エポック数の探索、部分フルFTとの比較。
4. kanasimがKeyErrorになる区間の扱い(採点不能を減らす前処理・別指標の併用)。

## 再現方法

```bash
# GPU機で
uv sync && uv add peft accelerate jiwer
uv run python scripts/build_kana_dataset_v2.py --n 1000        # 音声再生成
uv run python scripts/train_kana_ctc.py --epochs 5 --batch 4 --grad-accum 2 \
    --out local/models/kana_ctc_v1                             # 学習+ホールドアウト評価
# スモークテスト: uv run python scripts/train_kana_ctc.py --limit 20 --epochs 1
```

学習済みLoRAアダプタと区間別の評価結果は `local/models/kana_ctc_v1/`(git管理外、`adapter_model.safetensors` / `holdout_results.json`)に出力される。
