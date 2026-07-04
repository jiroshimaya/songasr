"""新手法B2: 日本語カナ書き起こし用wav2vec2モデルに英語音声をそのまま入力する。

jiroshimaya/songasr の scripts/run_kana_asr.py と同じアイデア。
AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana は日本語の
ひらがな・カタカナを出力するよう学習されたCTCモデル。CTCは1フレームごとに
何らかのラベルを出す非自己回帰モデルなので、Whisperのように文脈から
それらしい文を「創作」することができず、音響的特徴に忠実な書き起こしになりやすい。
"""

from __future__ import annotations

import functools
import re

import pyopenjtalk
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

MODEL_ID = "AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana"

_KATAKANA_RE = re.compile(r"[ア-ヴー]+")


@functools.lru_cache(maxsize=1)
def _load_model() -> tuple[Wav2Vec2ForCTC, AutoProcessor]:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


def text_to_katakana(text: str) -> str:
    phonemes = pyopenjtalk.g2p(text, kana=True)
    if isinstance(phonemes, list):
        phonemes = "".join(phonemes)
    return "".join(_KATAKANA_RE.findall(phonemes))


def wav2vec2_ja_transcribe(audio, sr: int = 16_000) -> dict[str, str]:
    model, processor = _load_model()
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    raw_text = processor.decode(pred_ids[0], skip_special_tokens=True)
    katakana = text_to_katakana(raw_text)
    return {"raw": raw_text, "kana": katakana}
