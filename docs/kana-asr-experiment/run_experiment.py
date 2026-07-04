"""baseline (wav2vec2-espeak+arpakana) vs wav2vec2-ja のカナASRを5文で比較し、
kanasimでリファレンスカナとの距離を採点する。"""

import json
from pathlib import Path

import soundfile as sf
from kanasim import create_kana_distance_calculator

from src.baseline import baseline_transcribe
from src.wav2vec2_ja import wav2vec2_ja_transcribe

AUDIO_DIR = Path(__file__).parent / "audio"

CASES = {
    "s1_moon": {
        "text": "Under the moon and the quiet sky so clear",
        "reference_kana": "アンダーザムーンアンドザクワイエットスカイソークリア",
    },
    "s2_hand": {
        "text": "I want to hold your hand tonight",
        "reference_kana": "アイウォントトゥホールドユアハンドトゥナイト",
    },
    "s3_strike": {
        "text": "Strike the drums and break the silence now",
        "reference_kana": "ストライクザドラムズアンドブレイクザサイレンスナウ",
    },
    "s4_contraction": {
        "text": "I can't believe it's already over",
        "reference_kana": "アイキャントビリーヴイッツオールレディオーバー",
    },
    "s5_love": {
        "text": "Can you feel the love in every heartbeat",
        "reference_kana": "キャンユーフィールザラヴインエヴリハートビート",
    },
}


def main() -> None:
    calc = create_kana_distance_calculator()
    results = {}
    for name, case in CASES.items():
        wav_path = AUDIO_DIR / f"{name}.wav"
        audio, sr = sf.read(str(wav_path), dtype="float32")
        print(f"=== {name}: {case['text']} ===")

        base = baseline_transcribe(audio, sr)
        base_kana = base["kana"]
        base_dist = calc.calculate(base_kana, case["reference_kana"])
        print(f"baseline ipa: {base['ipa']}")
        print(f"baseline arpabet: {base['arpabet']}")
        print(f"baseline kana: {base_kana}  (dist={base_dist:.2f})")

        w2v = wav2vec2_ja_transcribe(audio, sr)
        w2v_kana = w2v["kana"]
        w2v_dist = calc.calculate(w2v_kana, case["reference_kana"])
        print(f"wav2vec2-ja raw: {w2v['raw']}")
        print(f"wav2vec2-ja kana: {w2v_kana}  (dist={w2v_dist:.2f})")

        print(f"reference: {case['reference_kana']}")
        print()

        results[name] = {
            "text": case["text"],
            "reference_kana": case["reference_kana"],
            "baseline": {**base, "distance": base_dist},
            "wav2vec2_ja": {**w2v, "distance": w2v_dist},
        }

    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"saved: {out_path}")

    avg_base = sum(r["baseline"]["distance"] for r in results.values()) / len(results)
    avg_w2v = sum(r["wav2vec2_ja"]["distance"] for r in results.values()) / len(results)
    print(f"\naverage kanasim distance: baseline={avg_base:.2f}  wav2vec2-ja={avg_w2v:.2f}")


if __name__ == "__main__":
    main()
