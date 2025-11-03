import sys
from faster_whisper import WhisperModel

# CLI引数から入力ファイルを取得
if len(sys.argv) < 2:
    print("使用方法: python run_kana_whisper_asr.py <入力音声ファイル>")
    sys.exit(1)

input_file = sys.argv[1]

# モデル（multilingual系）をロード
model = WhisperModel("small")  # 任意のサイズでOK

# 抑制するトークンIDを読み込み
with open("local/tokens_to_ignore.txt", "r", encoding="utf-8") as f:
    suppress_tokens = [int(x) for x in f.read().split() if x.strip().isdigit()]

# 推論（英語歌唱をカナ発音で書き起こし）
segments, info = model.transcribe(
    input_file,
    language="ja",
    task="transcribe",
    initial_prompt="カタカナで書いてください",  # よりシンプルに
    suppress_tokens=suppress_tokens,   # 英語歌唱の場合は抑制しない
    beam_size=5,
    temperature=0.3,  # 少し上げる
    # 英語歌唱認識のため、より寛容な設定
    log_prob_threshold=None,
    compression_ratio_threshold=3.0,
    no_speech_threshold=0.6,
)

# テキスト結合
text = "".join(seg.text for seg in segments)
print(text)