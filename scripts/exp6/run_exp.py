#!/usr/bin/env python3
"""issue #6: gold 653クリップでの精度改善実験 (GPU機で実行)。

ステージ制・レジューム可能 (idごとにJSONL追記、既存idはスキップ):
  finetuned        : #3 LoRAモデルで原音声(mix)を認識
  whisper          : faster-whisper large-v3 で原音声を英語書き起こし
  demucs           : htdemucs --two-stems=vocals でボーカル分離 (CLI呼び出し)
  baseline_vocals  : baseline (wav2vec2-espeak→ARPABET→カナ) を分離vocalに適用 (GPU)
  finetuned_vocals : LoRAモデルを分離vocalに適用
  whisper_vocals   : whisper large-v3 を分離vocalに適用
  score            : 全手法を kanasim (スペース除去) で採点し summary を出力

使い方 (リポジトリルートで):
  PYTHONPATH=src .venv/bin/python local/exp6/run_exp.py <stage> [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "local/gold_dataset"
OUT = ROOT / "local/exp6"
DEMUCS_DIR = OUT / "demucs"
TARGET_SR = 16_000

SMALL_KANA = set("ャュョァィゥェォヮ")


def load_manifest() -> list[dict]:
    rows = []
    with (GOLD / "manifest.jsonl").open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_done(path: Path) -> dict[str, dict]:
    done = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                done[r["id"]] = r
    return done


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def read_audio_16k(path: Path):
    import librosa
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio


def vocals_path(clip_id: str) -> Path:
    return DEMUCS_DIR / "htdemucs" / clip_id / "vocals.wav"


# ---------- finetuned (LoRA) ----------

def make_ft_transcriber():
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    from songasr.kana_wav2vec2_ja import MODEL_ID, text_to_katakana

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(base, str(ROOT / "local/models/kana_ctc_v1"))
    model.eval().to("cuda")

    def transcribe(audio) -> dict:
        if len(audio) < 8000:
            audio = np.pad(audio, (0, 8000 - len(audio)))
        inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to("cuda")).logits
        ids = torch.argmax(logits, dim=-1)
        raw = processor.batch_decode(ids)[0]
        return {"raw": raw, "kana": text_to_katakana(raw)}

    return transcribe


# ---------- baseline on GPU ----------

def make_baseline_transcriber():
    import numpy as np
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    from songasr.kana_baseline import MODEL_ID, ipa_to_arpabet

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval().to("cuda")

    def transcribe(audio) -> dict:
        from arpakana import arpabet_to_kana

        if len(audio) < 8000:
            audio = np.pad(audio, (0, 8000 - len(audio)))
        inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to("cuda")).logits
        pred_ids = torch.argmax(logits, dim=-1)
        ipa_text = processor.batch_decode(pred_ids)[0]
        toks = ipa_to_arpabet(re.split(r"\s+", ipa_text.strip()))
        kana = arpabet_to_kana(toks, unknown="") if toks else ""
        return {"ipa": ipa_text, "kana": kana}

    return transcribe


# ---------- whisper ----------

def make_whisper_transcriber():
    from faster_whisper import WhisperModel

    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    def transcribe(path: Path) -> dict:
        segments, _info = model.transcribe(
            str(path), language="en", vad_filter=False, beam_size=5
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return {"english": text}

    return transcribe


# ---------- stages ----------

def run_model_stage(stage: str, limit: int | None) -> None:
    rows = load_manifest()
    out_path = OUT / f"{stage}.jsonl"
    # エラーレコードは再試行対象にする (OOM等の一時要因を想定)
    done = {k: v for k, v in load_done(out_path).items() if "error" not in v}

    use_vocals = stage.endswith("_vocals")
    kind = stage.replace("_vocals", "")

    if kind == "finetuned":
        fn = make_ft_transcriber()
        audio_input = True
    elif kind == "baseline":
        fn = make_baseline_transcriber()
        audio_input = True
    elif kind == "whisper":
        fn = make_whisper_transcriber()
        audio_input = False
    else:
        raise ValueError(stage)

    n = 0
    for row in rows:
        if limit is not None and n >= limit:
            break
        cid = row["id"]
        if cid in done:
            continue
        path = vocals_path(cid) if use_vocals else GOLD / row["audio_file"]
        if not path.exists():
            print(f"[{stage}] SKIP {cid}: {path} not found", flush=True)
            continue
        try:
            if audio_input:
                res = fn(read_audio_16k(path))
            else:
                res = fn(path)
        except Exception as e:  # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}"}
        rec = {"id": cid, **res}
        append_jsonl(out_path, rec)
        n += 1
        if n % 25 == 0:
            print(f"[{stage}] {n} processed (last: {cid})", flush=True)
    print(f"[{stage}] done, {n} new records -> {out_path}", flush=True)


def run_demucs(limit: int | None) -> None:
    # demucs CLI は torchaudio→torchcodec の I/O が壊れている(libnvrtc.so.13要求)ため、
    # Python API + soundfile/librosa で I/O を自前実装する。
    import librosa
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    rows = load_manifest()
    todo = [row for row in rows if not vocals_path(row["id"]).exists()]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        print("[demucs] nothing to do", flush=True)
        return

    model = get_model("htdemucs")
    model.eval().to("cuda")
    vocals_idx = model.sources.index("vocals")
    model_sr = model.samplerate  # 44100

    print(f"[demucs] separating {len(todo)} clips", flush=True)
    for i, row in enumerate(todo, 1):
        audio = read_audio_16k(GOLD / row["audio_file"])
        up = librosa.resample(audio, orig_sr=TARGET_SR, target_sr=model_sr)
        wav = torch.from_numpy(up).float().unsqueeze(0).repeat(2, 1)  # (2, samples)
        # demucs.separate と同じ入力正規化
        ref = wav.mean(0)
        std = ref.std() + 1e-8
        wav = (wav - ref.mean()) / std
        with torch.no_grad():
            out = apply_model(model, wav[None].to("cuda"), device="cuda")[0]
        vocals = (out[vocals_idx].cpu() * std + ref.mean()).mean(0).numpy()
        down = librosa.resample(vocals, orig_sr=model_sr, target_sr=TARGET_SR)
        vp = vocals_path(row["id"])
        vp.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(vp), down, TARGET_SR, subtype="PCM_16")
        if i % 25 == 0:
            print(f"[demucs] {i}/{len(todo)} (last: {row['id']})", flush=True)
    print(f"[demucs] done, {len(todo)} clips", flush=True)


def run_whisper_song(limit: int | None) -> None:
    """全曲を large-v3 + word timestamps で転写し、単語列を曲ごとに保存する。

    per-clip 転写(whisperステージ)は3〜8秒の切片単独で文脈を失い
    whisper-small の全曲転写(english_sung)にすら負ける。実運用形は
    「全曲転写→単語をクリップ区間へマップ」なのでこちらを測る。
    """
    from faster_whisper import WhisperModel

    stage_file = ROOT / "local/gold_stage/stage.jsonl"
    songs = [json.loads(x) for x in stage_file.read_text().splitlines() if x.strip()]
    if limit is not None:
        songs = songs[:limit]
    out_path = OUT / "whisper_song.jsonl"
    done = load_done(out_path)

    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    for sng in songs:
        if sng["slug"] in done:
            continue
        segments, _info = model.transcribe(
            str(ROOT / "local/gold_stage" / sng["wav"]),
            language="en", word_timestamps=True, vad_filter=False, beam_size=5,
        )
        words = []
        for seg in segments:
            for w in seg.words or []:
                words.append({"s": round(w.start, 3), "e": round(w.end, 3),
                              "w": w.word})
        append_jsonl(out_path, {"id": sng["slug"], "words": words})
        print(f"[whisper_song] {sng['slug']}: {len(words)} words", flush=True)
    print(f"[whisper_song] done -> {out_path}", flush=True)


def clip_words_text(words: list[dict], start: float, end: float,
                    pad: float = 0.2) -> str:
    picked = [w["w"] for w in words
              if w["e"] > start - pad and w["s"] < end + pad]
    return "".join(picked).strip()


# ---------- scoring ----------

def norm_kana(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def mora_len(kana: str) -> int:
    return sum(1 for ch in norm_kana(kana) if ch not in SMALL_KANA)


def english_to_kana_e2k(text: str, c2k) -> str:
    words = re.findall(r"[a-zA-Z']+", (text or "").lower())
    kanas = []
    for w in words:
        w2 = w.replace("'", "")
        if not w2:
            continue
        try:
            kanas.append(c2k(w2))
        except Exception:  # noqa: BLE001
            continue
    return " ".join(kanas)


def run_score() -> None:
    from kanasim import create_kana_distance_calculator

    calc = create_kana_distance_calculator()

    def dist(pred: str, ref: str):
        p, r = norm_kana(pred), norm_kana(ref)
        if not p or not r:
            return None, "empty"
        try:
            return calc.calculate(p, r), None
        except KeyError as e:
            return None, f"KeyError:{e}"

    try:
        from e2k import C2K

        c2k = C2K()
    except ImportError:
        c2k = None
        print("[score] WARNING: e2k not installed; whisper kana conversion skipped")

    rows = load_manifest()
    stage_preds: dict[str, dict[str, dict]] = {}
    for stage in ["finetuned", "whisper", "baseline_vocals",
                  "finetuned_vocals", "whisper_vocals"]:
        stage_preds[stage] = load_done(OUT / f"{stage}.jsonl")

    methods = [
        "baseline_mix", "finetuned_mix", "whisper_e2k_mix", "e2k_english_sung",
        "e2k_english_ref", "whisper_song_e2k",
        "baseline_vocals", "finetuned_vocals", "whisper_e2k_vocals",
    ]
    song_words = {r["id"]: r.get("words", [])
                  for r in load_done(OUT / "whisper_song.jsonl").values()}

    results = []
    for row in rows:
        cid = row["id"]
        gold = row["kana"]
        rec: dict = {"id": cid, "song": row["song"], "gold": gold,
                     "gold_mora": mora_len(gold),
                     "duration_sec": row.get("duration_sec")}

        preds: dict[str, str | None] = {
            "baseline_mix": row.get("baseline_pred"),
            "finetuned_mix": (stage_preds["finetuned"].get(cid) or {}).get("kana"),
            "baseline_vocals": (stage_preds["baseline_vocals"].get(cid) or {}).get("kana"),
            "finetuned_vocals": (stage_preds["finetuned_vocals"].get(cid) or {}).get("kana"),
        }
        if c2k is not None:
            preds["e2k_english_sung"] = english_to_kana_e2k(row.get("english_sung", ""), c2k)
            # oracle: 正解英語歌詞をe2kでカナ化 = テキスト→カナ経路のASR誤りゼロ上限
            preds["e2k_english_ref"] = english_to_kana_e2k(row.get("english_ref", ""), c2k)
            wm = (stage_preds["whisper"].get(cid) or {}).get("english")
            preds["whisper_e2k_mix"] = english_to_kana_e2k(wm, c2k) if wm else None
            wv = (stage_preds["whisper_vocals"].get(cid) or {}).get("english")
            preds["whisper_e2k_vocals"] = english_to_kana_e2k(wv, c2k) if wv else None
            slug = cid.rsplit("_", 1)[0]
            if slug in song_words:
                txt = clip_words_text(song_words[slug],
                                      row["start_sec"], row["end_sec"])
                preds["whisper_song_e2k"] = english_to_kana_e2k(txt, c2k)
                rec["whisper_song_en"] = txt

        for m in methods:
            p = preds.get(m)
            if p is None:
                continue
            d, err = dist(p, gold)
            rec[f"{m}_pred"] = p
            rec[f"{m}_dist"] = d
            if err:
                rec[f"{m}_err"] = err
        results.append(rec)

    res_path = OUT / "results.jsonl"
    with res_path.open("w") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- summary ----
    def summarize(method: str, rows_: list[dict]) -> dict | None:
        pairs = [(r[f"{method}_dist"], r["gold_mora"]) for r in rows_
                 if r.get(f"{method}_dist") is not None and r["gold_mora"] > 0]
        n_total = sum(1 for r in rows_ if f"{method}_pred" in r)
        if not pairs:
            return None
        dpm = sum(d / m for d, m in pairs) / len(pairs)
        return {"method": method, "n_evaluated": n_total,
                "n_scored": len(pairs),
                "mean_dist": round(sum(d for d, _ in pairs) / len(pairs), 2),
                "d_per_mora": round(dpm, 3)}

    summary: dict = {"overall": [], "per_song": {}, "head_to_head": {}}
    for m in methods:
        s = summarize(m, results)
        if s:
            summary["overall"].append(s)

    songs = sorted({r["song"] for r in results})
    for song in songs:
        srows = [r for r in results if r["song"] == song]
        summary["per_song"][song] = [s for m in methods
                                     if (s := summarize(m, srows))]

    # baseline_mix との直接対決 (両方採点可のクリップのみ)
    for m in methods:
        if m == "baseline_mix":
            continue
        wins = ties = losses = 0
        for r in results:
            a, b = r.get(f"{m}_dist"), r.get("baseline_mix_dist")
            if a is None or b is None:
                continue
            if a < b:
                wins += 1
            elif a == b:
                ties += 1
            else:
                losses += 1
        tot = wins + ties + losses
        if tot:
            summary["head_to_head"][m] = {
                "vs": "baseline_mix", "n": tot, "wins": wins,
                "losses": losses, "ties": ties,
                "win_rate": round(wins / tot, 3)}

    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["head_to_head"], ensure_ascii=False, indent=2))
    print(f"[score] results -> {res_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=[
        "finetuned", "whisper", "demucs", "baseline_vocals",
        "finetuned_vocals", "whisper_vocals", "whisper_song", "score"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "demucs":
        run_demucs(args.limit)
    elif args.stage == "whisper_song":
        run_whisper_song(args.limit)
    elif args.stage == "score":
        run_score()
    else:
        run_model_stage(args.stage, args.limit)


if __name__ == "__main__":
    main()
