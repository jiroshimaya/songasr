"""
split_audio_by_segments.py のテスト
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pydub import AudioSegment

from scripts.split_audio_by_segments import (
    load_whisper_segments,
    sanitize_filename,
    split_audio_by_segments,
)


class TestLoadWhisperSegments:
    def test_正常系_有効なJSONファイルで結果(self, tmp_path: Path):
        # テスト用のJSONファイルを作成
        test_segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello world"},
            {"start": 5.0, "end": 10.0, "text": "This is a test"},
        ]
        json_file = tmp_path / "test.json"
        with json_file.open("w", encoding="utf-8") as f:
            json.dump(test_segments, f)

        # テスト実行
        result = load_whisper_segments(json_file)

        # 検証
        assert result == test_segments

    def test_異常系_存在しないファイルで例外(self):
        non_existent_file = Path("non_existent.json")

        with pytest.raises(FileNotFoundError):
            load_whisper_segments(non_existent_file)


class TestSanitizeFilename:
    def test_正常系_基本的な文字列で結果(self):
        assert sanitize_filename("Hello World") == "Hello_World"

    def test_正常系_特殊文字除去で結果(self):
        assert sanitize_filename("Hello/World\\Test?") == "HelloWorldTest"

    def test_正常系_長い文字列で切り詰めで結果(self):
        long_text = "a" * 100
        result = sanitize_filename(long_text)
        assert len(result) <= 50

    def test_正常系_空文字列でデフォルト名で結果(self):
        assert sanitize_filename("") == "segment"

    def test_正常系_連続するスペースでアンダースコアで結果(self):
        assert sanitize_filename("Hello    World   Test") == "Hello_World_Test"


class TestSplitAudioBySegments:
    @patch("scripts.split_audio_by_segments.AudioSegment.from_file")
    def test_正常系_音声分割で結果(self, mock_from_file, tmp_path: Path):
        # モック設定
        mock_audio = Mock()
        mock_audio.__len__ = Mock(return_value=10000)  # 10秒の音声
        mock_audio.__getitem__ = Mock(return_value=mock_audio)
        mock_audio.export = Mock()
        mock_from_file.return_value = mock_audio

        # テストデータ
        wav_file = tmp_path / "test.wav"
        wav_file.touch()  # ファイルを作成
        output_dir = tmp_path / "output"

        segments = [
            {"start": 0.0, "end": 2.5, "text": "First segment"},
            {"start": 2.5, "end": 5.0, "text": "Second segment"},
        ]

        # テスト実行
        split_audio_by_segments(wav_file, segments, output_dir)

        # 検証
        assert output_dir.exists()
        mock_from_file.assert_called_once_with(str(wav_file))
        assert mock_audio.export.call_count == 2

    @patch("scripts.split_audio_by_segments.AudioSegment.from_file")
    def test_正常系_終了時間調整で結果(self, mock_from_file, tmp_path: Path):
        # モック設定
        mock_audio = Mock()
        mock_audio.__len__ = Mock(return_value=20000)  # 20秒の音声
        mock_audio.__getitem__ = Mock(return_value=mock_audio)
        mock_audio.export = Mock()
        mock_from_file.return_value = mock_audio

        # テストデータ（短いセグメントと次のセグメントが近い場合）
        wav_file = tmp_path / "test.wav"
        wav_file.touch()
        output_dir = tmp_path / "output"

        segments = [
            {"start": 0.0, "end": 1.0, "text": "Short segment"},  # +1秒で2.0秒
            {
                "start": 1.5,
                "end": 3.0,
                "text": "Next segment",
            },  # 次の開始が1.5秒なので1.5秒まで
            {"start": 5.0, "end": 7.0, "text": "Last segment"},  # +1秒で8.0秒
        ]

        # テスト実行
        split_audio_by_segments(wav_file, segments, output_dir)

        # 検証：セグメント分割が呼ばれた回数と引数
        assert mock_audio.__getitem__.call_count == 3
        # 最初のセグメント: 0ms-1500ms（次のセグメント開始時間1.5秒まで）
        mock_audio.__getitem__.assert_any_call(slice(0, 1500))
        # 2番目のセグメント: 1500ms-4000ms（+1秒で4.0秒）
        mock_audio.__getitem__.assert_any_call(slice(1500, 4000))
        # 3番目のセグメント: 5000ms-8000ms（+1秒で8.0秒）
        mock_audio.__getitem__.assert_any_call(slice(5000, 8000))

    def test_異常系_存在しない音声ファイルで例外(self, tmp_path: Path):
        non_existent_file = tmp_path / "non_existent.wav"
        output_dir = tmp_path / "output"
        segments = [{"start": 0.0, "end": 5.0, "text": "Test"}]

        with pytest.raises(FileNotFoundError):
            split_audio_by_segments(non_existent_file, segments, output_dir)


class TestMainScript:
    def test_正常系_CLIスクリプト実行で結果(self, tmp_path: Path):
        # テスト用のJSONファイルを作成
        test_segments = [
            {"start": 0.0, "end": 2.0, "text": "Test segment"},
        ]
        json_file = tmp_path / "test.json"
        with json_file.open("w", encoding="utf-8") as f:
            json.dump(test_segments, f)

        # テスト用の音声ファイルを作成（1秒の無音）
        wav_file = tmp_path / "test.wav"
        silent_audio = AudioSegment.silent(duration=3000)  # 3秒
        silent_audio.export(str(wav_file), format="wav")

        output_dir = tmp_path / "output"

        # スクリプトを実行
        result = subprocess.run(
            [
                "uv",
                "run",
                "scripts/split_audio_by_segments.py",
                "--whisper-json",
                str(json_file),
                "--wav-file",
                str(wav_file),
                "--output-dir",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            check=False,
        )

        # 検証
        assert result.returncode == 0
        assert output_dir.exists()
        output_files = list(output_dir.glob("*.wav"))
        assert len(output_files) == 1

    def test_異常系_不正な引数で例外(self):
        result = subprocess.run(
            ["uv", "run", "scripts/split_audio_by_segments.py", "--invalid-arg"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            check=False,
        )

        assert result.returncode != 0
