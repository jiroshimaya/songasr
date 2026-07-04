"""疑似ラベル用データセットの元になる英語短文(歌詞ふうのオリジナル文)を生成する。

実在の楽曲の歌詞をそのまま使うと著作権上の懸念があるため、ポップス歌詞に
よくあるパターン(主語+動詞+目的語/形容詞、機能語、縮約形など)をテンプレートで
組み合わせて、オリジナルの短文を機械的に生成している。
kana_asr_experimentで使った5文(s1〜s5, TTS実験と同一)は継続性のためそのまま含める。
"""

from __future__ import annotations

import itertools
import random

# TTS実験(kana_asr_experiment)で使った5文。継続比較のためそのまま含める。
ORIGINAL_FIVE = [
    "Under the moon and the quiet sky so clear",
    "I want to hold your hand tonight",
    "Strike the drums and break the silence now",
    "I can't believe it's already over",
    "Can you feel the love in every heartbeat",
]

SUBJECTS = [
    "I", "we", "you", "she", "he", "my heart", "your love", "the night",
    "this moment", "our story", "the rain", "the stars", "my soul",
]

VERBS_OBJ = [
    "hold you close", "feel the rhythm", "chase the dawn", "break these chains",
    "dance in the dark", "whisper your name", "burn like fire", "fade away slowly",
    "shine so bright", "run through the rain", "carry this dream", "wait for the sunrise",
    "remember it all", "let it go now", "find our way home", "reach for the sky",
]

CONNECTORS = [
    "and", "but", "so", "while", "when", "because", "until",
]

TAGS = [
    "tonight", "forever", "again", "somehow", "right now", "in the end",
    "one more time", "under the stars", "before the dawn", "without a sound",
]

CONTRACTION_LINES = [
    "I can't stop thinking about you",
    "We won't back down tonight",
    "It's already written in the stars",
    "You're the only one I need",
    "I'll never let you go",
    "Don't you dare walk away",
    "She's gone but I remember",
    "We've been waiting for this",
    "I'm falling for you again",
    "They don't know what we know",
]

CONSONANT_CLUSTER_LINES = [
    "Strength and struggle build the story",
    "Crash through the crowd of strangers",
    "Splash of light across the street",
    "Twist and shout until the morning",
    "Scratch the surface, find the truth",
    "Thunder strikes beneath the bridge",
    "Grasp the moment, don't let go",
    "Flash of hope in the middle of the storm",
]


def generate_template_sentences(n: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    combos = list(itertools.product(SUBJECTS, VERBS_OBJ, TAGS))
    rng.shuffle(combos)

    sentences: list[str] = []
    seen = set()
    for subj, verb_obj, tag in combos:
        line = f"{subj} {verb_obj} {tag}".strip()
        line = line[0].upper() + line[1:]
        if line not in seen:
            seen.add(line)
            sentences.append(line)
        if len(sentences) >= n:
            break

    # 接続詞でつないだやや長めの文もいくつか混ぜる
    combos2 = list(itertools.product(SUBJECTS, VERBS_OBJ, CONNECTORS, VERBS_OBJ))
    rng.shuffle(combos2)
    for subj, v1, conn, v2 in combos2:
        if len(sentences) >= n:
            break
        line = f"{subj} {v1} {conn} {v2}"
        line = line[0].upper() + line[1:]
        if line not in seen:
            seen.add(line)
            sentences.append(line)

    return sentences[:n]


def build_corpus(target_total: int = 120) -> list[str]:
    """ORIGINAL_FIVE + 縮約形 + 子音クラスタ + テンプレート生成文、で構成する。"""
    corpus = list(ORIGINAL_FIVE)
    corpus.extend(CONTRACTION_LINES)
    corpus.extend(CONSONANT_CLUSTER_LINES)

    remaining = max(0, target_total - len(corpus))
    corpus.extend(generate_template_sentences(remaining))

    # 重複除去(順序維持)
    seen = set()
    unique = []
    for s in corpus:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique[:target_total]


if __name__ == "__main__":
    for i, s in enumerate(build_corpus(), 1):
        print(f"{i:03d}: {s}")
