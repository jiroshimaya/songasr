"""Whisperのvocabからカタカナ・ひらがな・句読点以外のトークンIDを洗い出す。

jiroshimaya/songasr の scripts/download_whisper_tokens.py と同じロジック。
"""

from __future__ import annotations

import base64
import functools
import re
from pathlib import Path

import requests

TIKTOKEN_URL = (
    "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken"
)
CACHE_DIR = Path(__file__).parent.parent / "local"
TIKTOKEN_PATH = CACHE_DIR / "multilingual.tiktoken"

_KANA_OR_PUNCT_RE = re.compile(r"^[぀-ヿ、。ー]+$")


def _download_tiktoken() -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    if not TIKTOKEN_PATH.exists():
        resp = requests.get(TIKTOKEN_URL, timeout=30)
        resp.raise_for_status()
        TIKTOKEN_PATH.write_bytes(resp.content)
    return TIKTOKEN_PATH


def _load_tokens(path: Path) -> list[tuple[str, int]]:
    tokens = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        b64_token, token_id = line.split()
        decoded = base64.b64decode(b64_token).decode("utf-8", errors="ignore")
        tokens.append((decoded, int(token_id)))
    return tokens


@functools.lru_cache(maxsize=1)
def get_non_kana_token_ids() -> list[int]:
    """カタカナ・ひらがな・句読点以外のトークンIDのリスト(=suppress_tokensに渡す)。"""
    path = _download_tiktoken()
    tokens = _load_tokens(path)
    return [tid for text, tid in tokens if not _KANA_OR_PUNCT_RE.match(text)]
