#!/usr/bin/env python3
"""issue #3: TTS疑似ラベル(v2, 1000文)で英語歌唱→カナ用CTCモデルをLoRAファインチューニングする。

ベースモデルは AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana
(1BパラメータのXLS-R CTC)。data/kana_dataset_v2/manifest.jsonl の (音声, カナ) ペアを
教師データに、LoRA + 学習可能なCTCヘッドだけを更新する(ベース重みはbf16 autocastで凍結)。
これによりRTX 4060 Ti (16GB)でも1Bモデルの学習が現実的に収まる。

学習データは弱教師あり(baseline疑似ラベル)であることに注意(docs/kana-asr-experiment/
DATASET_NOTES.md 参照)。真の評価は学習に使わない実歌唱27区間ホールドアウト
(scripts/build_song_corpus.py の SEGMENTS)で kanasim 距離を測り、
baseline(平均88.15)/wav2vec2-ja直接推論(平均105.13)と比較して行う。

使い方:
  uv run python scripts/train_kana_ctc.py --limit 20 --epochs 1   # スモークテスト
  uv run python scripts/train_kana_ctc.py                         # 本番(全1000件)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
from kanasim import create_kana_distance_calculator
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    Wav2Vec2ForCTC,
)

# scripts/ を import path に追加し、ホールドアウト評価の部品(27区間の定義・音声読込・
# kanasim安全計算)を build_song_corpus から再利用する(import しても main() は走らない)。
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_song_corpus as bsc  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_kana_ctc")

MODEL_ID = "AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana"
TARGET_SR = 16_000
MANIFEST = Path("data/kana_dataset_v2/manifest.jsonl")
AUDIO_DIR = Path("local/kana_dataset_v2/audio")
DEFAULT_OUT = Path("local/models/kana_ctc_v1")

_KATAKANA_RE = re.compile(r"[ァ-ヴー]+")


def to_katakana(text: str) -> str:
    """ひらがなをカタカナへ寄せ、カタカナ(+長音)以外を落とす。参照ラベルがカタカナのため。"""
    kata = "".join(
        chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text
    )
    return "".join(_KATAKANA_RE.findall(kata))


# --------------------------------------------------------------------------- #
# データ
# --------------------------------------------------------------------------- #
def load_records(limit: int | None) -> list[dict[str, Any]]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"manifest not found: {MANIFEST}")
    records = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    records = [r for r in records if r.get("kana")]  # 空ラベルは除外(v2では0件のはず)
    if limit is not None:
        records = records[:limit]
    return records


class KanaDataset(Dataset):
    """manifestの1行 → {input_values, labels} を返す。音声は都度ロード(1000件なら十分軽い)。"""

    def __init__(self, records: list[dict[str, Any]], processor: Any) -> None:
        self.records = records
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        path = AUDIO_DIR / rec["audio_file"]
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        input_values = self.processor(
            audio, sampling_rate=TARGET_SR
        ).input_values[0]
        labels = self.processor.tokenizer(rec["kana"]).input_ids
        return {"input_values": input_values, "labels": labels}


@dataclass
class DataCollatorCTCWithPadding:
    """入力とラベルを別々に動的パディング。ラベルのpadは-100にしてCTC lossから除外する。"""

    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=True, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


# --------------------------------------------------------------------------- #
# ホールドアウト評価(実歌唱27区間, kanasim)
# --------------------------------------------------------------------------- #
def evaluate_holdout(model: Any, processor: Any, device: torch.device) -> dict[str, Any]:
    song_audio = bsc.ensure_song()
    calc = create_kana_distance_calculator()
    model.eval()
    records = []
    for i, (start, end, text, ref_kana) in enumerate(bsc.SEGMENTS, 1):
        clip = bsc.extract(song_audio, start, end)
        inputs = processor(clip, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        raw = processor.batch_decode(pred_ids)[0]
        kana = to_katakana(raw)
        dist = bsc.safe_distance(calc, kana, ref_kana)
        records.append({
            "index": i, "text": text, "reference_kana": ref_kana,
            "pred_kana": kana, "distance": dist,
        })
        logger.info("[holdout %02d/%d] pred=%s (%s) ref=%s", i, len(bsc.SEGMENTS), kana, dist, ref_kana)
    valid = [r["distance"] for r in records if r["distance"] is not None]
    mean = sum(valid) / len(valid) if valid else None
    return {"records": records, "mean_distance": mean, "n_valid": len(valid)}


# --------------------------------------------------------------------------- #
# メイン
# --------------------------------------------------------------------------- #
def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """評価時に全語彙ロジットを溜め込まず、argmax済みのID列だけ保持してメモリを抑える。"""
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def build_compute_metrics(processor: Any):
    def compute_metrics(pred: Any) -> dict[str, float]:
        pred_ids = pred.predictions  # preprocess_logits_for_metricsでargmax済み
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        # 空参照はjiwerがエラーになるので除外
        pairs = [(p, r) for p, r in zip(pred_str, label_str, strict=False) if r]
        if not pairs:
            return {"cer": 1.0}
        preds, refs = zip(*pairs, strict=False)
        return {"cer": jiwer.cer(list(refs), list(preds))}

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="使う件数(スモーク用)")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--skip-holdout", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # 語彙カバレッジのfail-fastチェック(拡張不要なはずだが念のため)
    records = load_records(args.limit)
    vocab = processor.tokenizer.get_vocab()
    label_chars = {c for r in records for c in r["kana"]}
    oov = sorted(c for c in label_chars if c not in vocab)
    if oov:
        raise ValueError(f"ラベルにOOV文字あり(語彙拡張が必要): {''.join(oov)}")
    logger.info("records=%d, label chars=%d, OOV=0", len(records), len(label_chars))

    random.shuffle(records)
    val_size = min(args.val_size, max(1, len(records) // 5))
    val_records, train_records = records[:val_size], records[val_size:]
    logger.info("train=%d val=%d", len(train_records), len(val_records))

    train_ds = KanaDataset(train_records, processor)
    val_ds = KanaDataset(val_records, processor)
    collator = DataCollatorCTCWithPadding(processor)

    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model.freeze_feature_encoder()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        modules_to_save=["lm_head"],
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    # wav2vec2は入力埋め込みを持たないためenable_input_require_gradsは使えない。
    # gradient checkpointingはuse_reentrant=Falseを使うのでこのトリックは不要
    # (LoRA/lm_headの学習可能パラメータはcheckpoint区間内にあり勾配が流れる)。

    training_args = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        report_to=[],
        label_names=["labels"],
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=build_compute_metrics(processor),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        processing_class=processor,
    )

    logger.info("=== 学習開始 ===")
    trainer.train()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    processor.save_pretrained(str(args.out))
    logger.info("モデルを保存: %s", args.out)

    if not args.skip_holdout:
        logger.info("=== ホールドアウト(実歌唱27区間)評価 ===")
        result = evaluate_holdout(model, processor, device)
        out_json = args.out / "holdout_results.json"
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        logger.info(
            "ホールドアウト平均kanasim距離: %s (n_valid=%d) — 比較: baseline=88.15 / wav2vec2-ja=105.13",
            f"{result['mean_distance']:.2f}" if result["mean_distance"] is not None else "N/A",
            result["n_valid"],
        )
        logger.info("詳細を保存: %s", out_json)


if __name__ == "__main__":
    main()
