"""新手法B1: Whisperを language="ja" で強制し、カタカナ以外のトークンを
suppress_tokens で禁止することで、英語音声をそのままカナ化する。

jiroshimaya/songasr の scripts/run_kana_whisper_asr.py と同じアイデア。
Whisperは本来「どの言語で喋っているか」を検出してから書き起こすが、
language="ja" を強制すると日本語の音韻体系で音声を解釈しようとする。
そこにカタカナ以外のトークンを全面禁止すると、英語の音をカタカナの音韻に
無理やり当てはめた書き起こし=空耳的な結果が得られる。
"""

from __future__ import annotations

import functools

import numpy as np
import whisper

from .kana_tokens import get_non_kana_token_ids

MODEL_NAME = "small"


@functools.lru_cache(maxsize=1)
def _load_model():
    # GPUは他プロセスに占有されていることが多いためCPU固定で実行する
    return whisper.load_model(MODEL_NAME, device="cpu")


def whisper_kana_transcribe(audio: np.ndarray) -> str:
    model = _load_model()
    suppress_tokens = get_non_kana_token_ids()
    result = model.transcribe(
        audio.astype(np.float32),
        language="ja",
        task="transcribe",
        suppress_tokens=suppress_tokens,
        beam_size=5,
        condition_on_previous_text=False,
    )
    return result.get("text", "").strip()
