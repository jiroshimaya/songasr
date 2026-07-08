"""e2k出力カナを gold の「歌える空耳」様式に近づける縮約ルール (issue #7)。

contract_words(words_en, kanas) -> kanas
単語ごとの英語表記とe2kカナを受け取り、縮約後のカナ列を返す。
ルールは train 12曲のみで設計し、test 6曲で汎化を評価すること
(train曲のマイニング結果: scripts/exp7/mine_patterns.py)。

環境変数 CONTRACT_RULES にカンマ区切りでルール名を指定すると
そのサブセットだけを適用する (アブレーション用)。既定は全ルール。
"""

from __future__ import annotations

import os
import re

# 高頻度機能語の直接マップ (train曲のgoldで縮約・有声化が確認できたもの)
FUNCTION_MAP = {
    "but": "バ",
    "and": "アン",
    "that": "ザッ",
    "the": "ダ",
    "is": "イズ",
    "was": "ワズ",
    "as": "アズ",
    "his": "ヒズ",
    "has": "ハズ",
    "these": "ジーズ",
    "those": "ゾーズ",
    "now": "ナウ",
    "know": "ノウ",
    "because": "ビカズ",
    "it": "イッ",
    "at": "アッ",
    "get": "ゲッ",
    "got": "ガッ",
    "what": "ワッ",
    "let": "レッ",
    "not": "ナッ",
    "just": "ジャス",
    "want": "ウォン",
    "of": "オブ",
}

# (ルール名, パターン, 置換) — 単語単位のカナに順に適用する正規表現
GENERIC_RULES: list[tuple[str, re.Pattern, str]] = [
    # 語末 ットス/ッツ → ツ (it's→イッツ→イツ 系)
    ("ttsu", re.compile(r"ットス$|ッツ$|トス$"), "ツ"),
    # 語末の ット/ッド → 全落ち (バット→バ)
    ("tto", re.compile(r"ット$|ッド$"), ""),
    # 長音・ン・イ・ウ の後の語末 ト/ド は落ちる (ライト→ライ, アンド→アン)
    ("t_after_long", re.compile(r"(?<=[ーンイウ])[トド]$"), ""),
    # 語末 ング → ン (フラッディング→フラッディン)
    ("ngu", re.compile(r"ング$"), "ン"),
    # 語末閉鎖音の破裂を落とす (ック/ップ→ッ)
    ("kku", re.compile(r"ッ[クプ]$"), "ッ"),
    # 長音+語末ル → 長音のみ (オール→オー, フィール→フィー)
    ("ru", re.compile(r"(?<=ー)ル$"), ""),
    # 3モーラ以上の語の語末長音を短縮 (オンリー→オンリ)
    ("final_choon", re.compile(r"(?<=..)ー$"), ""),
    # 除外済みルール (leave-one-outアブレーションで有害と判明):
    #   ム$→ン (フリーダム→フリーダン), 語中長音短縮 (ギーブ→ギブ)
]


def _enabled() -> set[str] | None:
    env = os.environ.get("CONTRACT_RULES")
    if env is None:
        return None  # 全ルール
    return {x.strip() for x in env.split(",") if x.strip()}


def contract_word(en: str, kana: str, enabled: set[str] | None) -> str:
    if not kana:
        return kana
    if (enabled is None or "funcmap" in enabled) and en in FUNCTION_MAP:
        return FUNCTION_MAP[en]
    for name, pat, repl in GENERIC_RULES:
        if enabled is not None and name not in enabled:
            continue
        kana = pat.sub(repl, kana)
    return kana


def contract_words(words_en: list[str], kanas: list[str]) -> list[str]:
    enabled = _enabled()
    return [contract_word(en, k, enabled) for en, k in zip(words_en, kanas)]
