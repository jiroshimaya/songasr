"""旧手法: wav2vec2 (espeak IPA) -> ipapy -> ARPABET -> arpakana でカナ化する。

前回のQiita記事 (https://qiita.com/shimajiroxyz/items/37c1e8d309403165e33c) の
再現実装。facebook/wav2vec2-lv-60-espeak-cv-ft はespeakのIPA音素セットで
学習された多言語音素認識モデルで、デコード結果はスペース区切りのIPA音素列になる。
"""

from __future__ import annotations

import collections
import collections.abc
import functools
import re

import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

# ipapy は Python3.12 で除去された collections.MutableSequence を要求するため
# collections.abc からのエイリアスをここで補っておく。
collections.MutableSequence = collections.abc.MutableSequence  # type: ignore[attr-defined]

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"

# espeak IPA -> ARPABET (CMUdict互換) の対応表。
# 標準的なARPABET-IPA対応表 (Wikipedia "ARPABET"記載の対応) をベースに、
# espeakが実際に出力する記号 (長音記号 ' ː '、シュワー系、Rカラーなど) を追加している。
IPA_TO_ARPABET: dict[str, str] = {
    # 母音
    "i": "IY", "iː": "IY", "ɪ": "IH", "ᵻ": "IH",
    "e": "EH", "ɛ": "EH", "eː": "EH",
    "æ": "AE",
    "ʌ": "AH", "ə": "AH", "ɐ": "AH",
    "ɑ": "AA", "ɑː": "AA", "aː": "AA",
    "ɒ": "AO", "ɔ": "AO", "ɔː": "AO",
    "ʊ": "UH",
    "u": "UW", "uː": "UW",
    "ɜ": "ER", "ɜː": "ER", "ɚ": "ER", "ɝ": "ER",
    "eɪ": "EY",
    "aɪ": "AY",
    "ɔɪ": "OY",
    "oʊ": "OW", "əʊ": "OW", "o": "OW",
    "aʊ": "AW",
    # 子音
    "p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "ɡ": "G", "g": "G",
    "tʃ": "CH", "dʒ": "JH",
    "f": "F", "v": "V", "θ": "TH", "ð": "DH",
    "s": "S", "z": "Z", "ʃ": "SH", "ʒ": "ZH",
    "h": "HH", "m": "M", "n": "N", "ŋ": "NG",
    "l": "L", "ɫ": "L",
    "r": "R", "ɹ": "R", "ɾ": "R",
    "j": "Y", "w": "W",
}


def ipa_to_arpabet(ipa_tokens: list[str], unknown: str = "") -> list[str]:
    """espeak IPA音素トークン列をARPABET音素列に変換する(未対応は無視)。"""
    out = []
    for tok in ipa_tokens:
        tok = tok.strip()
        if not tok:
            continue
        # ストレス記号 (ˈˌ) や長さ以外の余分な発音区別記号を除去してから引く
        cleaned = tok.replace("ˈ", "").replace("ˌ", "")
        mapped = IPA_TO_ARPABET.get(cleaned) or IPA_TO_ARPABET.get(cleaned.rstrip("ː"))
        if mapped:
            out.append(mapped)
        elif unknown:
            out.append(unknown)
    return out


@functools.lru_cache(maxsize=1)
def _load_model() -> tuple[Wav2Vec2ForCTC, AutoProcessor]:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor  # GPUは他プロセスに占有されているためCPUのまま使う


def audio_to_ipa(audio, sr: int = 16_000) -> str:
    model, processor = _load_model()
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def baseline_transcribe(audio, sr: int = 16_000) -> dict[str, str]:
    """旧手法でカナ化する。中間結果 (IPA, ARPABET) も返す。"""
    from arpakana import arpabet_to_kana

    ipa_text = audio_to_ipa(audio, sr)
    ipa_tokens = re.split(r"\s+", ipa_text.strip())
    arpabet_tokens = ipa_to_arpabet(ipa_tokens)
    # 未知音素は "?" を残さずカナに含めない (kanasimの距離表に存在しないため)
    kana = arpabet_to_kana(arpabet_tokens, unknown="") if arpabet_tokens else ""
    return {
        "ipa": ipa_text,
        "arpabet": " ".join(arpabet_tokens),
        "kana": kana,
    }
