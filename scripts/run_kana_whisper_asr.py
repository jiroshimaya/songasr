import sys
from pathlib import Path

import whisper

TOKENS_PATH = Path("local/tokens_to_ignore.txt")


def load_suppress_tokens(file_path: Path) -> list[int]:
    tokens: list[int] = []
    if not file_path.exists():
        raise FileNotFoundError(f"suppressトークンファイルが見つかりません: {file_path}")
    for token_str in file_path.read_text(encoding="utf-8").split():
        try:
            tokens.append(int(token_str))
        except ValueError:
            continue
    return tokens


def main() -> None:
    if len(sys.argv) < 2:
        print("使用方法: python run_kana_whisper_asr.py <入力音声ファイル>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"入力ファイルが見つかりません: {input_path}")
        sys.exit(1)

    suppress_tokens = load_suppress_tokens(TOKENS_PATH)

    model = whisper.load_model("large-v2")

    result = model.transcribe(
        str(input_path),
        language="ja",
        task="transcribe",
        suppress_tokens=suppress_tokens,
        beam_size=3,
    )

    print(result.get("text", ""))

if __name__ == "__main__":
    main()