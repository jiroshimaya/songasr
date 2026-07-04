"""Generate TTS audio for the test sentences (gTTS -> mp3 -> 16kHz mono wav array, no ffmpeg)."""

import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from gtts import gTTS

SENTENCES = {
    "s1_moon": "Under the moon and the quiet sky so clear",
    "s2_hand": "I want to hold your hand tonight",
    "s3_strike": "Strike the drums and break the silence now",
    "s4_contraction": "I can't believe it's already over",
    "s5_love": "Can you feel the love in every heartbeat",
}

OUT_DIR = Path(__file__).parent / "audio"
OUT_DIR.mkdir(exist_ok=True)

TARGET_SR = 16_000


def synthesize(name: str, text: str) -> Path:
    mp3_path = OUT_DIR / f"{name}.mp3"
    wav_path = OUT_DIR / f"{name}.wav"
    if not mp3_path.exists():
        tts = gTTS(text=text, lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        mp3_path.write_bytes(buf.getvalue())
        print(f"saved mp3: {mp3_path}")

    # decode mp3 via soundfile (libsndfile 1.2+ supports mp3 natively, no ffmpeg)
    audio, sr = sf.read(str(mp3_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    # pad 0.3s silence both ends
    pad = int(0.3 * TARGET_SR)
    audio = np.pad(audio, pad)
    sf.write(str(wav_path), audio, TARGET_SR, subtype="PCM_16")
    print(f"saved wav: {wav_path} ({len(audio) / TARGET_SR:.2f}s)")
    return wav_path


if __name__ == "__main__":
    for name, text in SENTENCES.items():
        synthesize(name, text)
