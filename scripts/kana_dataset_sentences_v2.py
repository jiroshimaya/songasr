"""疑似ラベル用データセット v2 の元になる英語短文を生成する(約1000文規模)。

v1 (kana_dataset_sentences.py, 120文) は1種類のテンプレート
(主語+動詞句+タグ)が中心で構造的な多様性が乏しかった。
v2では以下の観点で多様性を意図的に増やしている:

- 手書きのシード文(約140文): 子音クラスタ(str/spl/scr/thr/spr/ngth等)、
  二重母音が目立つ単語(aɪ/aʊ/ɔɪ/eɪ/oʊ)、母音の多い言い回し、
  縮約形(I'm/don't/can't/won't/it's/we'll/you're等)、
  2〜3語の短いフレーズから12〜14語の長いフレーズまで、長さのばらつきを
  意図して用意した。
- 複数の異なる文構造テンプレート(平叙文、命令文、疑問文、倒置、
  接続詞で繋いだ複文など)を使い、単一パターンの反復にならないようにした。
- 依然として実在の楽曲の歌詞はそのまま使わず、オリジナル文のみ。
"""

from __future__ import annotations

import itertools
import random

# --------------------------------------------------------------------------
# 1. 手書きシード文: 特定の音韻パターンを狙って作った短文
# --------------------------------------------------------------------------

CONSONANT_CLUSTER_SEEDS = [
    "Strength through the struggle, we stand tall",
    "Splash of starlight on the street",
    "Scratch beneath the surface of the truth",
    "Thunder cracks across the twisted sky",
    "Spring into the story we forgot",
    "Strange shadows stretch beyond the bridge",
    "Twist the truth until it breaks",
    "Grasp the strings of what remains",
    "Crash through the crowd, don't look back",
    "Splinters of glass beneath the stars",
    "Strict rhythm, sharp and unforgiving",
    "Screech of tires on a sprawling road",
    "Straight through the storm we sprint",
    "Scrape the sky with broken wings",
    "Spread your arms and trust the fall",
]

DIPHTHONG_SEEDS = [
    "I know why the night feels so alive",
    "My eyes shine brighter than the fire",
    "Now the crowd shouts loud and proud",
    "Wait for the day the light arrives",
    "The joy in your voice is my prize",
    "Fly high, don't let the sky decide",
    "Time flies by while the flame stays low",
    "I found my way home in the afterglow",
    "Down by the sound of the crowded town",
    "Slow motion, golden road, we go",
    "White light on the tide, we ride",
    "I tried to hide, but the fire won't die",
    "Around the world, the sound will grow",
    "Shine on, shine on, until the dawn",
    "Boy, oh boy, the noise destroys the quiet",
]

VOWEL_HEAVY_SEEDS = [
    "I owe you a year of easy days",
    "A quiet ocean, a lonely idea",
    "You are the aura I adore",
    "Oh, I owe an era to you",
    "Away, away, our idea will stay",
    "Eerie air over an empty area",
    "A beautiful aria for you and I",
    "I ache to see you again",
]

CONTRACTION_SEEDS = [
    "I'm not afraid of what's ahead",
    "Don't you dare forget my name",
    "Can't stop the feeling running through my veins",
    "Won't back down till the morning comes",
    "It's already too late to turn around",
    "We'll rise above the noise tonight",
    "You're the reason I believe in stars",
    "They've never seen a love like ours",
    "Shouldn't have said goodbye so soon",
    "Isn't it strange how memories fade",
    "Hasn't anyone told you you're enough",
    "I'd give it all just to see you smile",
    "We're standing at the edge of everything",
    "That's the last thing I remember",
    "I've been waiting for a sign",
    "Wouldn't change a single thing we did",
    "Couldn't find the words to say goodbye",
    "Didn't know that love could hurt this good",
]

SHORT_PHRASES = [
    "Hold on tight",
    "Let it burn",
    "Stay with me",
    "Fade away",
    "Rise and shine",
    "Break the silence",
    "Chase the light",
    "Set me free",
    "Never look back",
    "Feel alive",
    "Dance tonight",
    "Come home",
    "Say my name",
    "Take my hand",
    "Run away",
]

LONG_PHRASES = [
    "Somewhere between the silence and the sound of your voice, I find my peace",
    "When the city sleeps and the neon lights flicker, we are still awake",
    "I've carried this feeling since the day you walked into my life",
    "Every road we take leads back to the place where it all began",
    "Underneath the weight of everything we lost, a small light still remains",
    "If you listen closely, you can hear the echo of a promise kept",
    "Long after the music stops, the memory keeps on playing in my head",
    "We built a home from broken pieces and called it something beautiful",
    "There's a version of tonight where neither of us ever says goodbye",
    "Somewhere past the mountains, past the rivers, there's a place we used to know",
]

THEME_SEEDS = {
    "night": [
        "The city lights flicker like a heartbeat in the dark",
        "Underneath the moonlight, we forget the time",
        "Stars are falling, but I'm not afraid",
    ],
    "dance": [
        "Move your body to the rhythm of the drums",
        "We spin until the room becomes a blur",
        "Every step we take rewrites the story",
    ],
    "rain": [
        "The rain keeps falling on an empty street",
        "Thunder rolls in like it owns the sky",
        "Puddles reflect the neon of the town",
    ],
    "fire": [
        "Sparks are flying off the edge of everything",
        "We burn brighter when the world goes cold",
        "The flame remembers every word we said",
    ],
    "freedom": [
        "Open roads and windows down, we're finally free",
        "No chains can hold the sound of a wild heart",
        "We're flying higher than the doubts they gave us",
    ],
    "heartbreak": [
        "Pieces of a heart scattered on the floor",
        "I keep replaying every word you never said",
        "Some wounds heal quiet, some just stay",
    ],
    "hope": [
        "There's a light at the end of every storm",
        "Hold on, morning always finds a way",
        "Even broken wings remember how to fly",
    ],
    "memory": [
        "Old photographs fade but the feeling stays",
        "I still hear your laughter in an empty room",
        "Some memories play on repeat, like a favorite song",
    ],
}


def _flatten_theme_seeds() -> list[str]:
    out = []
    for lines in THEME_SEEDS.values():
        out.extend(lines)
    return out


HAND_WRITTEN_SEEDS: list[str] = (
    CONSONANT_CLUSTER_SEEDS
    + DIPHTHONG_SEEDS
    + VOWEL_HEAVY_SEEDS
    + CONTRACTION_SEEDS
    + SHORT_PHRASES
    + LONG_PHRASES
    + _flatten_theme_seeds()
)

# --------------------------------------------------------------------------
# 2. 複数の文構造テンプレートによる組み合わせ生成(構造的多様性の確保)
# --------------------------------------------------------------------------

SUBJECTS = [
    "I", "we", "you", "she", "he", "my heart", "your love", "the night",
    "this moment", "our story", "the rain", "the stars", "my soul",
    "the city", "this silence", "your voice", "the fire inside",
]

VERB_PHRASES = [
    "hold you close", "feel the rhythm", "chase the dawn", "break these chains",
    "dance in the dark", "whisper your name", "burn like fire", "fade away slowly",
    "shine so bright", "run through the rain", "carry this dream", "wait for the sunrise",
    "remember it all", "let it go now", "find our way home", "reach for the sky",
    "scream into the void", "drift through the static", "cling to the echo",
    "spiral out of control", "settle into silence", "climb over the ruins",
]

TAGS = [
    "tonight", "forever", "again", "somehow", "right now", "in the end",
    "one more time", "under the stars", "before the dawn", "without a sound",
    "like it's the last time", "when no one's watching", "just to feel something",
]

CONNECTORS = ["and", "but", "so", "while", "when", "because", "until", "even though"]

QUESTION_STARTS = [
    "Do you", "Can you", "Won't you", "Will we ever", "How could I",
    "Why does it", "What if we", "Where did the",
]

QUESTION_BODIES = [
    "feel this fire burning inside", "remember how it used to be",
    "stay a little longer tonight", "make it through the storm",
    "find our way back home", "let the silence say it all",
    "forget the way we started", "hold on when it hurts this much",
]

IMPERATIVE_VERBS = [
    "hold", "chase", "break", "burn", "whisper", "remember", "let go of",
    "reach for", "run from", "dance through", "carry", "scream into",
]

IMPERATIVE_OBJECTS = [
    "the night", "this moment", "your fear", "the silence", "our story",
    "the fire", "tomorrow", "every memory", "the storm inside", "this feeling",
]


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def generate_svo(rng: random.Random) -> str:
    subj, verb_obj, tag = rng.choice(SUBJECTS), rng.choice(VERB_PHRASES), rng.choice(TAGS)
    return _cap(f"{subj} {verb_obj} {tag}")


def generate_compound(rng: random.Random) -> str:
    subj = rng.choice(SUBJECTS)
    v1, conn, v2 = rng.choice(VERB_PHRASES), rng.choice(CONNECTORS), rng.choice(VERB_PHRASES)
    return _cap(f"{subj} {v1} {conn} {v2}")


def generate_question(rng: random.Random) -> str:
    start, body = rng.choice(QUESTION_STARTS), rng.choice(QUESTION_BODIES)
    return f"{start} {body}?"


def generate_imperative(rng: random.Random) -> str:
    verb, obj, tag = rng.choice(IMPERATIVE_VERBS), rng.choice(IMPERATIVE_OBJECTS), rng.choice(TAGS)
    return _cap(f"{verb} {obj} {tag}")


def generate_two_clause(rng: random.Random) -> str:
    subj1, v1 = rng.choice(SUBJECTS), rng.choice(VERB_PHRASES)
    subj2, v2 = rng.choice(SUBJECTS), rng.choice(VERB_PHRASES)
    return _cap(f"{subj1} {v1}, {subj2} {v2}")


GENERATORS = [
    generate_svo, generate_compound, generate_question,
    generate_imperative, generate_two_clause,
]


def generate_template_sentences(n: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []
    # 各テンプレートを均等に回して構造の偏りを避ける
    gen_cycle = itertools.cycle(GENERATORS)
    attempts = 0
    max_attempts = n * 50
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        gen = next(gen_cycle)
        line = gen(rng)
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def build_corpus(target_total: int = 1000, seed: int = 7) -> list[str]:
    corpus: list[str] = list(HAND_WRITTEN_SEEDS)
    remaining = max(0, target_total - len(corpus))
    corpus.extend(generate_template_sentences(remaining, seed=seed))

    seen: set[str] = set()
    unique: list[str] = []
    for s in corpus:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique[:target_total]


if __name__ == "__main__":
    lines = build_corpus()
    print(f"total: {len(lines)}")
    for i, s in enumerate(lines[:30], 1):
        print(f"{i:04d}: {s}")
