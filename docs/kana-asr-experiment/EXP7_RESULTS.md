# テキスト→カナ経路の改善: 空耳縮約 + large-v3セグメント評価 (issue #7)

[#7](https://github.com/jiroshimaya/songasr/issues/7) の実施記録。#6 で確定した方針「英語ASR→テキスト→カナ化」の実行編。コードは `scripts/exp7/`、成果物は GPU機の `local/exp7/`。

**結論: 目標を全て達成。** 実ASR経路（large-v3全曲転写→e2k→縮約ルール）で test曲 D/mora **3.54**（目標≤4.5、ストレッチ≤4.2）。oracle+縮約は **2.98**（目標≤3.5）。#6 時点の最良 5.49 から約2.0の改善。

## 設定

- 指標・採点は #6 と同じ（D/mora、スペース除去kanasim、チャンス≈7.6）。
- **train/test 分割は曲単位**: slugソートで idx%3==2 の6曲が test（dive, galway_girl, never_enough, save_myself, the_other_side, what_do_i_know）。縮約ルールの設計・アブレーションは train 12曲のみで実施。
- 評価プロトコル: ASR自身のセグメントを yougaku 英語行にファジーマッチ（≥0.6）し、対応カナを正解とする。english_sung（whisper-small由来のgoldクリップ）と同型。

## 結果

| 経路 | all | train | test |
|---|--:|--:|--:|
| english_sung(whisper-small)→e2k素朴 (#6最良) | 5.49 | 5.28 | 5.91 |
| english_sung→e2k→**縮約** | 4.19 | 3.92 | 4.70 |
| large-v3セグメント→e2k素朴 | 4.87 | 4.97 | 4.61 |
| **large-v3セグメント→e2k→縮約** | **3.58** | 3.60 | **3.54** |
| oracle: 正解英語歌詞→e2k素朴 | 4.07 | 4.10 | 4.01 |
| oracle→e2k→**縮約** | **2.89** | 2.84 | **2.98** |

- 寄与の分解（test）: ASRを small→large-v3(セグメント単位) で −1.3、縮約で −1.1。**縮約はASR誤り入りテキストでも壊れず効く**。
- large-v3 のカバレッジ: マッチしたセグメント971件、yougaku行 506/979 (51.7%)。
- 残る oracle との差 ≈0.6 が ASRテキスト誤り、oracle+縮約 2.98 と 0 の差が縮約でまだ吸収できていない様式差。

## 空耳縮約ルール (`scripts/exp7/soramimi_contract.py`)

train曲の gold↔e2k 差分マイニング（`mine_patterns.py`）から設計。単語単位で適用:

1. **機能語マップ**（but→バ, and→アン, the→ダ, is→イズ, now→ナウ 等24語）
2. 語末 ットス/ッツ→ツ、語末 ット/ッド→削除（バット→バ）
3. 長音・ン・イ・ウ の後の語末 ト/ド 削除（ライト→ライ, アンド→アン）※単独で最大の寄与(−0.34)
4. 語末 ング→ン、語末 ック/ップ→ッ、長音後の語末 ル 削除（オール→オー）
5. 3モーラ以上の語末長音短縮（オンリー→オンリ）

leave-one-out アブレーション（train）で **ム→ン と語中長音短縮は有害**と判明し除外。採用セットは `funcmap,ttsu,tto,t_after_long,ngu,kku,ru,final_choon`。

出力例（test曲）:

```
en:   To the other side
gold: トゥーディーアーザーサイ
pred: トーダアザサイ            (D/mora 3.15 ≒ 中央値帯)
```

## 残課題（次の一手）

- **ワーストケースはマッチング起因**: ASRセグメントが正解行より長い範囲をカバーすると余剰語が距離を悪化させる（D/mora>10の外れ値の主因）。セグメント分割/行マッチの改善で数値はまだ下がる。
- e2k の系統誤り: could→カルド、to→トー、like you→ライクユー（gold ライキュー; t+y/k+y の口蓋化未対応）など。機能語マップ拡充 or 縮約の学習化（gold 653ペアでの軽量seq2seq）が次の伸びしろ。
- カバレッジ51.7%の残り = マッチ≥0.6に届かない行。ASR誤りと歌詞リファレンスのズレ両方を含む。用途（歌える歌詞シート生成）ではセグメント単位の出力で十分だが、行カバレッジを上げるならアラインメント側の工夫が必要。

## 再現方法

```bash
# GPU機のリポジトリルートで (e2k が必要: .venv/bin/pip install e2k)
export LD_LIBRARY_PATH=$(find $PWD/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name lib -type d | tr '\n' ':')
PY="PYTHONPATH=src:scripts:scripts/exp7 .venv/bin/python scripts/exp7/run_exp7.py"
eval "$PY transcribe"                       # large-v3 全曲転写 (GPU, ~10分)
eval "$PY eval --source large --contract"   # 本命経路の評価
eval "$PY eval --source ref --contract"     # oracle+縮約
CONTRACT_RULES=... で縮約ルールのアブレーション、
scripts/exp7/mine_patterns.py で差分パターンの再マイニング。
```
