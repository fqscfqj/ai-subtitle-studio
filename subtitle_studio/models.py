from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict, Optional, Protocol


Segment = Dict[str, Any]
WordToken = Dict[str, Any]
ProgressCallback = Callable[[str, int, str], None]


class TaskCancelled(Exception):
    pass


@dataclass
class WhisperProviderSettings:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "whisper-1"


@dataclass
class TranscriptionSettings:
    provider: str = "whisper_openai_compatible"
    language_mode: str = "auto"
    language: str = ""
    timestamp_granularity: str = "segment"
    context_bias: str = ""
    thread_count: int = 3
    max_retries: int = 3
    whisper: WhisperProviderSettings = field(default_factory=WhisperProviderSettings)


@dataclass
class TranslationSettings:
    mode: str = "none"
    model: str = "gpt-4o-mini"
    target_language: str = "zh"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    bilingual_srt: bool = True
    keep_original_srt: bool = False
    allow_subtitle_import: bool = True
    subtitle_translation_thread_count: int = 3
    thinking_enabled: bool = False
    reasoning_effort: str = "high"
    temperature: float = 0.2
    chunk_size: int = 40


@dataclass
class SegmentationSettings:
    enabled: bool = False
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_words_per_window: int = 180
    thinking_enabled: bool = False
    reasoning_effort: str = "high"


@dataclass
class OutputSettings:
    mode: str = "source"
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "subtitles")
    save_srt: bool = True
    save_lrc: bool = True
    save_txt: bool = True
    save_json: bool = False


@dataclass
class VadSettings:
    enabled: bool = False
    min_speech_ms: int = 250
    min_silence_ms: int = 400
    speech_pad_ms: int = 200
    max_segment_seconds: int = 15 * 60
    threshold: float = 0.5


@dataclass
class AppSettings:
    ui_theme: str = "light"
    retry_base_delay: float = 2.0
    transcription: TranscriptionSettings = field(default_factory=TranscriptionSettings)
    translation: TranslationSettings = field(default_factory=TranslationSettings)
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    vad: VadSettings = field(default_factory=VadSettings)


@dataclass
class TaskState:
    task_id: str
    source_path: Path
    row: int
    status: str = "Queued"
    progress: int = 0
    message: str = "就绪"
    outputs: Dict[str, str] = field(default_factory=dict)
    priority: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    error_detail: str = ""


@dataclass
class TranscriptionRequest:
    source_path: Path
    audio_path: Path
    language_mode: str
    language: str
    timestamp_granularity: str
    context_bias: str


@dataclass
class TranscriptionResult:
    text: str
    segments: list[Segment]
    language: str
    raw_payload: Dict[str, Any]


@dataclass
class TranslationRequest:
    model: str
    target_language: str


@dataclass
class SegmentationRequest:
    model: str
    source_language: str = ""


class TranscriptionProvider(Protocol):
    def transcribe(
        self,
        request: TranscriptionRequest,
        progress_cb: Optional[Callable[[str], None]],
        cancel_event: Event,
    ) -> TranscriptionResult:
        ...


class TranslationProvider(Protocol):
    def translate_lines(
        self,
        lines: list[str],
        request: TranslationRequest,
        cancel_event: Event,
        parallel_workers: int = 1,
    ) -> list[str]:
        ...


class SegmentationProvider(Protocol):
    def segment_words(
        self,
        words: list[WordToken],
        request: SegmentationRequest,
        cancel_event: Event,
    ) -> list[tuple[int, int]]:
        ...

