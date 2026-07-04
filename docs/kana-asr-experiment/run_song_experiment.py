"""実際の歌唱音源(Pixabayのロイヤリティフリー楽曲)でbaseline/wav2vec2-jaを比較する。

音源はjiroshimaya/songasrのscripts/download_sample_song.pyや、
Qiita記事 https://qiita.com/shimajiroxyz/items/7e43427134330faa6731 で
使われているのと同じPixabayの無料音源
(https://cdn.pixabay.com/download/audio/2025/03/22/audio_f46c5fa5ad.mp3)。
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from kanasim import create_kana_distance_calculator

from src.baseline import baseline_transcribe
from src.wav2vec2_ja import wav2vec2_ja_transcribe

AUDIO_DIR = Path(__file__).parent / "audio"
SONG_WAV = AUDIO_DIR / "song_16k.wav"
TARGET_SR = 16_000

# whisper(英語モード)の書き起こしから選んだ代表5区間。
# s1はTTS実験のs1_moonとまったく同じ歌詞(=このPixabay音源が前回記事の
# 例文の出どころだった)なので、TTS音声 vs 本物の歌声の直接比較になる。
SEGMENTS = [
    {
        "name": "song1_moon",
        "start": 4.30, "end": 11.12,
        "text": "Under the moon and the quiet sky so clear",
        "reference_kana": "アンダーザムーンアンドザクワイエットスカイソークリア",
    },
    {
        "name": "song2_whisper",
        "start": 12.00, "end": 17.66,
        "text": "A whisper of hope is all I hold near",
        "reference_kana": "アウィスパーオブホープイズオールアイホールドニア",
    },
    {
        "name": "song3_breath",
        "start": 17.66, "end": 23.50,
        "text": "Every breath, every word",
        "reference_kana": "エヴリブレスエヴリワード",
    },
    {
        "name": "song4_hands",
        "start": 50.72, "end": 55.64,
        "text": "Hands raised high with the hearts in tears",
        "reference_kana": "ハンズレイズドハイウィズザハーツインティアーズ",
    },
    {
        "name": "song5_eternal",
        "start": 63.72, "end": 69.72,
        "text": "Oh eternal light, that shines so pure",
        "reference_kana": "オーエターナルライトザットシャインズソーピュア",
    },
]

PAD_BEFORE = 0.2
PAD_AFTER = 0.3


def extract_segment(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    s = max(0, int((start - PAD_BEFORE) * TARGET_SR))
    e = min(len(audio), int((end + PAD_AFTER) * TARGET_SR))
    return audio[s:e]


def main() -> None:
    song_audio, sr = sf.read(str(SONG_WAV), dtype="float32")
    assert sr == TARGET_SR
    calc = create_kana_distance_calculator()

    results = {}
    for seg in SEGMENTS:
        name = seg["name"]
        clip = extract_segment(song_audio, seg["start"], seg["end"])
        sf.write(str(AUDIO_DIR / f"{name}.wav"), clip, TARGET_SR, subtype="PCM_16")
        dur = len(clip) / TARGET_SR

        print(f"=== {name} ({dur:.2f}s): {seg['text']} ===")

        base = baseline_transcribe(clip, TARGET_SR)
        base_dist = calc.calculate(base["kana"], seg["reference_kana"]) if base["kana"] else None
        print(f"baseline kana: {base['kana']}  (dist={base_dist})")

        w2v = wav2vec2_ja_transcribe(clip, TARGET_SR)
        w2v_dist = calc.calculate(w2v["kana"], seg["reference_kana"]) if w2v["kana"] else None
        print(f"wav2vec2-ja kana: {w2v['kana']}  (dist={w2v_dist})")

        print(f"reference: {seg['reference_kana']}")
        print()

        results[name] = {
            "text": seg["text"],
            "duration_sec": dur,
            "reference_kana": seg["reference_kana"],
            "baseline": {**base, "distance": base_dist},
            "wav2vec2_ja": {**w2v, "distance": w2v_dist},
        }

    out_path = Path(__file__).parent / "results_song.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")

    dists_base = [r["baseline"]["distance"] for r in results.values() if r["baseline"]["distance"] is not None]
    dists_w2v = [r["wav2vec2_ja"]["distance"] for r in results.values() if r["wav2vec2_ja"]["distance"] is not None]
    print(f"\naverage kanasim distance (song): baseline={sum(dists_base)/len(dists_base):.2f}  "
          f"wav2vec2-ja={sum(dists_w2v)/len(dists_w2v):.2f}")


if __name__ == "__main__":
    main()
