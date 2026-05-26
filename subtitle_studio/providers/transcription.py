from __future__ import annotations

import base64
import importlib
import json
from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict, Optional

from ..http_client import HttpClient
from ..models import (
    MistralProviderSettings,
    Qwen3ASRProviderSettings,
    TranscriptionProvider,
    TranscriptionRequest,
    TranscriptionResult,
    WhisperProviderSettings,
)
from ..utils import detect_language_code, extract_segments, extract_text, normalize_response

_MISTRAL_IMPORT_ERROR: Exception | None = None
try:
    import mistralai.client as mistralai_client

    Mistral = getattr(mistralai_client, "Mistral", None)
    if Mistral is None:
        raise AttributeError("mistralai.client.Mistral is unavailable")
except Exception:
    try:
        mistralai_module = importlib.import_module("mistralai")
        Mistral = getattr(mistralai_module, "Mistral")
    except Exception as exc:
        Mistral = None
        _MISTRAL_IMPORT_ERROR = exc


def normalize_audio_transcriptions_url(base_url: str) -> str:
    url = base_url.strip()
    if not url:
        return "https://api.openai.com/v1/audio/transcriptions"
    if url.endswith("/audio/transcriptions"):
        return url
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return f"{url}/audio/transcriptions"


class MistralTranscriptionProvider(TranscriptionProvider):
    def __init__(self, settings: MistralProviderSettings) -> None:
        self.settings = settings

    def transcribe(
        self,
        request: TranscriptionRequest,
        progress_cb: Optional[Callable[[str], None]],
        cancel_event: Event,
    ) -> TranscriptionResult:
        if cancel_event.is_set():
            raise RuntimeError("转写前已取消")
        if Mistral is None:
            details = ""
            if _MISTRAL_IMPORT_ERROR is not None:
                details = f"（导入错误：{type(_MISTRAL_IMPORT_ERROR).__name__}: {_MISTRAL_IMPORT_ERROR}）"
            raise RuntimeError(f"缺少依赖：mistralai{details}")
        client = Mistral(api_key=self.settings.api_key)

        kwargs: Dict[str, Any] = {"model": self.settings.model}
        if request.timestamp_granularity != "none":
            kwargs["timestamp_granularities"] = [request.timestamp_granularity]
        elif request.language_mode == "manual" and request.language:
            kwargs["language"] = request.language
        if request.diarize:
            kwargs["diarize"] = True
        if request.context_bias:
            kwargs["context_bias"] = request.context_bias

        if progress_cb:
            progress_cb("正在调用 Mistral API")
        with request.audio_path.open("rb") as file_obj:
            response = client.audio.transcriptions.complete(
                file={"content": file_obj, "file_name": request.audio_path.name},
                **kwargs,
            )
        payload = normalize_response(response)
        return TranscriptionResult(
            text=extract_text(payload),
            segments=extract_segments(payload),
            language=detect_language_code(payload),
            raw_payload=payload,
        )


class WhisperOpenAICompatibleProvider(TranscriptionProvider):
    def __init__(self, settings: WhisperProviderSettings, http_client: Optional[HttpClient] = None) -> None:
        self.settings = settings
        self.http_client = http_client or HttpClient()

    def transcribe(
        self,
        request: TranscriptionRequest,
        progress_cb: Optional[Callable[[str], None]],
        cancel_event: Event,
    ) -> TranscriptionResult:
        if cancel_event.is_set():
            raise RuntimeError("转写前已取消")
        if progress_cb:
            progress_cb("正在调用 Whisper 接口")

        with request.audio_path.open("rb") as file_obj:
            audio_bytes = file_obj.read()

        endpoint = normalize_audio_transcriptions_url(self.settings.base_url)
        data = self._build_form_data(request, include_timestamps=request.timestamp_granularity != "none")
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        files = {
            "file": (
                request.audio_path.name,
                audio_bytes,
                self._guess_mime_type(request.audio_path),
            )
        }

        response = self.http_client.post_multipart(endpoint, data=data, files=files, headers=headers)
        if response.status_code >= 400 and data.get("timestamp_granularities[]"):
            error_text = response.text.lower()
            if "timestamp" in error_text or "granularit" in error_text:
                fallback_data = self._build_form_data(request, include_timestamps=False)
                response = self.http_client.post_multipart(endpoint, data=fallback_data, files=files, headers=headers)

        if response.status_code >= 400:
            raise RuntimeError(
                f"Whisper 兼容接口返回错误: HTTP {response.status_code} {response.text[:240]}"
            )
        payload = response.payload if isinstance(response.payload, dict) else {"text": response.text}

        segments = extract_segments(payload)
        text = extract_text(payload)
        language = detect_language_code(payload)
        return TranscriptionResult(
            text=text,
            segments=segments,
            language=language,
            raw_payload=payload,
        )

    def _build_form_data(self, request: TranscriptionRequest, include_timestamps: bool) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "model": self.settings.model,
            "response_format": "verbose_json",
        }
        if request.language_mode == "manual" and request.language:
            data["language"] = request.language
        if request.context_bias:
            data["prompt"] = request.context_bias
        if include_timestamps:
            data["timestamp_granularities[]"] = request.timestamp_granularity
        return data

    def _guess_mime_type(self, path: Path) -> str:
        suffix = path.suffix.lower()
        return {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
        }.get(suffix, "application/octet-stream")


_DASHSCOPE_OPENAI_COMPAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

_MIME_MAP = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
    ".aac": "audio/aac", ".flac": "audio/flac", ".ogg": "audio/ogg",
    ".opus": "audio/ogg", ".webm": "audio/webm", ".wma": "audio/x-ms-wma",
    ".avi": "video/avi", ".mkv": "video/x-matroska", ".mov": "video/quicktime",
    ".mp4": "video/mp4", ".flv": "video/x-flv", ".mpeg": "video/mpeg",
    ".amr": "audio/amr", ".aiff": "audio/aiff",
}


class Qwen3ASRProvider(TranscriptionProvider):
    def __init__(self, settings: Qwen3ASRProviderSettings, http_client: Optional[HttpClient] = None) -> None:
        self.settings = settings
        self.http_client = http_client or HttpClient(timeout_seconds=600)
        # qwen3-asr-flash-filetrans 不支持 OpenAI 兼容模式，自动修正
        if "filetrans" in self.settings.model:
            self.settings.model = "qwen3-asr-flash"

    def transcribe(
        self,
        request: TranscriptionRequest,
        progress_cb: Optional[Callable[[str], None]],
        cancel_event: Event,
    ) -> TranscriptionResult:
        if cancel_event.is_set():
            raise RuntimeError("转写前已取消")

        audio_path = request.audio_path
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 10:
            raise RuntimeError(
                f"Qwen3 ASR (qwen3-asr-flash) 单文件限制 10MB，当前文件 {file_size_mb:.1f}MB。"
                "请使用 VAD 预切分或将音频上传至公网后用 qwen3-asr-flash-filetrans。"
            )

        if progress_cb:
            progress_cb("正在读取音频文件")
        suffix = audio_path.suffix.lower()
        mime_type = _MIME_MAP.get(suffix, "application/octet-stream")
        audio_bytes = audio_path.read_bytes()
        b64_data = base64.b64encode(audio_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{b64_data}"

        if progress_cb:
            progress_cb("正在调用 Qwen3 ASR")
        payload = self._call_sync_api(data_url, request, cancel_event)

        if cancel_event.is_set():
            raise RuntimeError("转写已取消")

        text = self._extract_text(payload)
        segments = self._extract_segments(payload)
        language = detect_language_code(payload)
        return TranscriptionResult(
            text=text,
            segments=segments,
            language=language,
            raw_payload=payload,
        )

    def _call_sync_api(
        self,
        data_url: str,
        request: TranscriptionRequest,
        cancel_event: Event,
    ) -> Dict[str, Any]:
        messages: list[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": data_url},
                    }
                ],
            }
        ]

        body: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
        }

        asr_options: Dict[str, Any] = {}
        if request.language_mode == "manual" and request.language:
            asr_options["language"] = request.language
        if request.context_bias:
            asr_options["prompt"] = request.context_bias
        if asr_options:
            body["asr_options"] = asr_options

        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

        response = self.http_client.post_json(_DASHSCOPE_OPENAI_COMPAT_URL, body, headers)
        if response.status_code >= 400:
            raise RuntimeError(
                f"Qwen3 ASR 接口返回错误: HTTP {response.status_code} {response.text[:240]}"
            )
        payload = response.payload if isinstance(response.payload, dict) else {}

        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError(f"Qwen3 ASR 返回空结果: {response.text[:240]}")

        return payload

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return message.get("content", "")

    @staticmethod
    def _extract_segments(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
        choices = payload.get("choices", [])
        if not choices:
            return []
        message = choices[0].get("message", {})
        text = message.get("content", "")
        if not text:
            return []
        return [{"start": 0.0, "end": 0.0, "text": text}]


def summarize_empty_transcription_response(payload: Dict[str, Any], raw_text: str) -> str:
    if payload:
        keys = ", ".join(sorted(str(key) for key in payload.keys())[:8])
        if keys:
            return f" 响应字段: {keys}。"
    snippet = raw_text.strip().replace("\r", " ").replace("\n", " ")
    if snippet:
        return f" 原始响应片段: {snippet[:240]}"
    return " 请检查服务端日志，以及所选模型是否真的支持 OpenAI 兼容的 audio/transcriptions 输出。"
