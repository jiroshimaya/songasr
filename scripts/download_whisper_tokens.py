"""download_sample_song.py

サンプル楽曲をダウンロードしてffmpegを使わずに純PythonでWAV変換するスクリプト
音源: Pixabay提供のフリー音楽素材
"""

import base64
import logging
import re
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

URL = "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken"
OUT_TXT = Path("local/multilingual.tiktoken")
IGNORE_TOKENS_FILE = Path("local/tokens_to_ignore.txt")


# 全角ひらがなカタカナおよび句読点判定用の関数
def is_katakana_or_punctuation(text) -> bool:
    return bool(re.match(r"^[\u3000-\u30FF、。]+$", text))


# multilingual.tiktokenファイルを読み込み、デコードしてトークンIDを取得する
def load_tokens(file_path) -> list[tuple[str, int]]:
    tokens = []
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            base64_token, token_id = line.strip().split()
            decoded_token = base64.b64decode(base64_token).decode(
                "utf-8", errors="ignore"
            )
            tokens.append((decoded_token, int(token_id)))
    return tokens


# カタカナおよび句読点以外のトークンを無視するためのトークンIDリストを作成
def get_non_katakana_or_punctuation_tokens(tokens) -> list[int]:
    non_katakana_or_punctuation_tokens = []
    for token, token_id in tokens:
        if is_katakana_or_punctuation(token):
            print(f"Ignore token: '{token}' (ID: {token_id})")
        if not is_katakana_or_punctuation(token):  # カタカナおよび句読点でない場合
            non_katakana_or_punctuation_tokens.append(token_id)
    return non_katakana_or_punctuation_tokens


# トークンIDをファイルに書き出す
def save_tokens_to_file(tokens, output_file_path) -> None:
    with open(output_file_path, "w", encoding="utf-8") as file:
        for token_id in tokens:
            file.write(f"{token_id}\n")


def download() -> None:
    """ffmpegを使わずに純Pythonでサンプル楽曲をダウンロードしてWAV形式に変換する"""
    # ディレクトリを作成
    OUT_TXT.parent.mkdir(exist_ok=True)

    # 1. ダウンロード
    logger.info(f"Downloading from {URL}")
    response = requests.get(URL, stream=True, timeout=30)
    response.raise_for_status()

    with OUT_TXT.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(f"Downloaded: {OUT_TXT}")


def main():
    download()

    # トークンを読み込み
    tokens = load_tokens(OUT_TXT)
    logger.info(f"Loaded {len(tokens)} tokens from {OUT_TXT}")

    # カタカナおよび句読点以外のトークンIDを取得
    tokens_to_ignore = get_non_katakana_or_punctuation_tokens(tokens)
    logger.info(f"Identified {len(tokens_to_ignore)} tokens to ignore")

    # トークンIDをファイルに保存
    save_tokens_to_file(tokens_to_ignore, IGNORE_TOKENS_FILE)
    logger.info(f"Saved tokens to ignore to {IGNORE_TOKENS_FILE}")


if __name__ == "__main__":
    main()
