"""
音声認識の共通処理

このモジュールには音声認識に関する共通的な機能を含む：
- モデル設定
- 音声ファイルの読み込み
- モデルのセットアップ
- 音声認識処理（フィルター方式・変換方式）
- テキスト変換処理
"""

import logging
import re
from typing import Any

import librosa
import numpy as np
import pyopenjtalk
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

logger = logging.getLogger(__name__)

# モデル設定
MODEL_CONFIGS = {
    "andrewmcdowell": {
        "model_id": "AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana",
    },
    "reazon": {
        "model_id": "reazon-research/japanese-wav2vec2-base-rs35kh",
    },
}

TARGET_SR = 16_000

# GPU最適化設定
DEFAULT_BATCH_SIZE = 8  # GPUメモリに応じて調整
MAX_AUDIO_LENGTH_SEC = 30  # 長すぎる音声のトリム制限


def get_device_info() -> dict:
    """デバイス情報を取得

    Returns:
        デバイス情報の辞書
    """
    info = {
        "device_type": "cpu",
        "device_name": "CPU",
        "memory_total": 0,
        "memory_available": 0,
    }

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_id)

        info.update(
            {
                "device_type": "cuda",
                "device_name": props.name,
                "memory_total": props.total_memory,
                "memory_available": (
                    torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_allocated()
                ),
            }
        )

    return info


def optimize_model_for_inference(model: Wav2Vec2ForCTC, device: torch.device) -> Any:
    """推論用にモデルを最適化

    Args:
        model: 最適化するモデル
        device: 使用デバイス

    Returns:
        最適化されたモデル
    """
    # 評価モードに設定
    model.eval()

    # PyTorch 2.0+ のtorch.compileが使用可能な場合
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            logger.info("Applying torch.compile optimization for GPU")
            model = torch.compile(model, mode="reduce-overhead")  # type: ignore
        except Exception as e:
            logger.warning(f"torch.compile failed, falling back: {e}")

    # GPUの場合はhalf精度を試行（メモリ削減・高速化）
    if device.type == "cuda":
        try:
            model = model.half()
            logger.info("Using half precision (float16) for GPU inference")
        except Exception as e:
            logger.warning(f"Half precision failed, using float32: {e}")

    return model


def text_to_katakana(text: str) -> str:
    """テキストをpyopenjtalkでカタカナに変換し、カタカナのみを抽出

    Args:
        text: 変換対象のテキスト

    Returns:
        カタカナのみのテキスト
    """
    # 読み仮名を取得
    phonemes = pyopenjtalk.g2p(text, kana=True)

    # phonemesがstrであることを保証
    if isinstance(phonemes, list):
        phonemes = "".join(phonemes)

    # カタカナ（ア-ヴ）と長音記号（ー）のみを抽出
    katakana_pattern = r"[ア-ヴー]+"
    katakana_matches = re.findall(katakana_pattern, phonemes)
    return "".join(katakana_matches)


def load_audio_mono_16k(path: str, pad_sec: float = 0.5) -> np.ndarray:
    """音声ファイルを16kHzモノラルで読み込み、オプションでパディング追加

    Args:
        path: 音声ファイルのパス
        pad_sec: パディング秒数

    Returns:
        音声波形データ
    """
    wav, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    if pad_sec and pad_sec > 0:
        pad = int(pad_sec * TARGET_SR)
        wav = np.pad(wav, pad_width=pad)
    # クリッピング対策
    if np.max(np.abs(wav)) > 1.0:
        wav = wav / (np.max(np.abs(wav)) + 1e-8)
    return wav.astype(np.float32)


def load_and_pad_audios(
    audio_paths: list[str],
    pad_sec: float = 0.5,
) -> list[np.ndarray]:
    """複数の音声ファイルを読み込み、パディング付きで統一長にする

    Args:
        audio_paths: 音声ファイルのパスリスト
        pad_sec: パディング秒数

    Returns:
        音声波形データのリスト
    """
    audios = []
    for path in audio_paths:
        wav = load_audio_mono_16k(path, pad_sec=pad_sec)
        audios.append(wav)
    return audios


def setup_model_and_processor(
    model_name: str, device: torch.device, optimize: bool = True
) -> tuple[Any, AutoProcessor]:
    """モデルとプロセッサーをセットアップ

    Args:
        model_name: 使用するモデル名
        device: 使用するデバイス
        optimize: GPU最適化を適用するかどうか

    Returns:
        モデルとプロセッサーのタプル
    """
    config = MODEL_CONFIGS[model_name]
    model_id = config["model_id"]

    # デバイス情報を表示
    device_info = get_device_info()
    logger.info(f"Loading model: {model_id}")
    logger.info(f"Device: {device_info['device_name']} ({device_info['device_type']})")
    if device_info["device_type"] == "cuda":
        total_gb = float(device_info["memory_total"]) / (1024**3)
        available_gb = float(device_info["memory_available"]) / (1024**3)
        logger.info(f"GPU Memory: {available_gb:.1f}GB / {total_gb:.1f}GB available")

    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    # デバイスに移動
    model = model.to(device)  # type: ignore

    # GPU最適化を適用
    if optimize and device.type == "cuda":
        model = optimize_model_for_inference(model, device)

    return model, processor


def transcribe_audios(  # noqa: PLR0912
    audio_paths: list[str],
    model_name: str = "andrewmcdowell",
    device: torch.device | None = None,
    pad_sec: float = 0.5,
    batch_size: int | None = None,
) -> list[str]:
    """複数の音声ファイルをバッチで転写（GPU最適化）

    Args:
        audio_paths: 音声ファイルのパスリスト
        mode: 転写方式（"filter" または "conversion"）
        model_name: 使用するモデル名
        device: 使用するデバイス
        pad_sec: パディング秒数
        batch_size: バッチサイズ（Noneの場合は自動調整）
        return_raw: 生のASR出力も返すかどうか（conversionモードのみ）

    Returns:
        認識結果のリスト
    """
    if not audio_paths:
        return []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # バッチサイズの自動調整
    if batch_size is None:
        if device.type == "cuda":
            # GPU使用時はメモリに応じて調整
            device_info = get_device_info()
            memory_gb = float(device_info["memory_available"]) / (1024**3)
            if memory_gb > 8:
                batch_size = 16
            elif memory_gb > 4:
                batch_size = 8
            else:
                batch_size = 4
        else:
            batch_size = 2  # CPU時は小さめのバッチ

    logger.info(
        "Batch transcription: %s files, batch_size=%s",
        len(audio_paths),
        batch_size,
    )

    # モデルとプロセッサーの準備（1度だけ）
    model, processor = setup_model_and_processor(model_name, device, optimize=True)

    results = []

    # バッチ処理
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i : i + batch_size]

        logger.info(
            "Processing batch %s/%s",
            i // batch_size + 1,
            (len(audio_paths) + batch_size - 1) // batch_size,
        )

        # 音声読み込み
        batch_audios = load_and_pad_audios(batch_paths, pad_sec)

        # 長さを統一（パディング）
        max_length = max(len(audio) for audio in batch_audios)
        padded_audios = []
        for audio in batch_audios:
            if len(audio) < max_length:
                pad_width = max_length - len(audio)
                padded_audio = np.pad(audio, (0, pad_width), mode="constant")
            else:
                padded_audio = audio
            padded_audios.append(padded_audio)

        # バッチテンソルに変換
        batch_array = np.stack(padded_audios)

        # プロセッサーで処理
        inputs = processor(
            batch_array, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
        )

        # デバイスに移動
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(device)

        # バッチ推論
        with torch.inference_mode():
            logits = model(**inputs).logits  # [B, T, V]
            pred_ids = torch.argmax(logits.detach().float().cpu(), dim=-1)

        # デコード処理
        for j in range(len(batch_paths)):
            raw_text = processor.decode(pred_ids[j], skip_special_tokens=True)

            # pyopenjtalkでカタカナに変換
            katakana_only = text_to_katakana(raw_text)
            results.append(katakana_only)

        # GPU メモリクリア
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Kana ASR 実行スクリプト")
    parser.add_argument(
        "audio_files",
        nargs="+",
        help="入力音声ファイルのパス（複数指定可能）",
    )
    parser.add_argument(
        "--model",
        default="andrewmcdowell",
        choices=list(MODEL_CONFIGS.keys()),
        help="使用するモデル名",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="使用するデバイス（cpu または cuda）。未指定の場合は自動選択。",
    )
    parser.add_argument(
        "--pad-sec",
        type=float,
        default=0.5,
        help="音声読み込み時のパディング秒数（デフォルト: 0.5秒）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="バッチサイズ（未指定の場合は自動調整）",
    )

    args = parser.parse_args()

    # デバイス設定
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 音声認識実行
    results = transcribe_audios(
        audio_paths=args.audio_files,
        model_name=args.model,
        device=device,
        pad_sec=args.pad_sec,
        batch_size=args.batch_size,
    )
    for path, text in zip(args.audio_files, results, strict=False):
        print(f"{Path(path).name}: {text}")
