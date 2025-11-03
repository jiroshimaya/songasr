#!/usr/bin/env python3
"""download_sample_ja_sentence.py

日本語音声サンプルをGoogle Driveからダウンロードして保存するスクリプト
"""

import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google DriveのファイルID（view URLから抽出）
FILE_ID = "142aj-qFJOhoteWKqgRzvNoq02JbZIsaG"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
OUT_FILE = Path("local/sample_ja_sentence.wav")


def download_sample_ja_sentence() -> None:
    """Google Driveから日本語音声サンプルをダウンロードする"""
    # ディレクトリを作成
    OUT_FILE.parent.mkdir(exist_ok=True)

    logger.info(f"Downloading from Google Drive (file ID: {FILE_ID})")

    # Google Driveからのダウンロードは少し特殊な処理が必要な場合がある
    session = requests.Session()

    response = session.get(URL, stream=True, timeout=30)

    # Google Driveが大きなファイルの場合、確認画面を表示することがある
    # その場合のハンドリング
    if "virus scan warning" in response.text or "download anyway" in response.text:
        # 確認画面をバイパスするためのトークンを取得
        for line in response.text.split("\n"):
            if "confirm=" in line and "download" in line:
                # トークンを抽出してリダイレクトURL作成
                import re

                confirm_token = re.search(r"confirm=([^&]+)", line)
                if confirm_token:
                    confirm_url = f"{URL}&confirm={confirm_token.group(1)}"
                    response = session.get(confirm_url, stream=True, timeout=30)
                    break

    response.raise_for_status()

    # ファイルを保存
    with OUT_FILE.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

    logger.info(f"ダウンロード完了: {OUT_FILE}")

    # ファイルサイズを確認
    file_size = OUT_FILE.stat().st_size
    logger.info(
        f"ファイルサイズ: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)"
    )


if __name__ == "__main__":
    download_sample_ja_sentence()
