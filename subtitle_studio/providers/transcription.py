from __future__ import annotations

import json
import time
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

try:
    from mistralai import Mistral
    _MISTRAL_IMPORT_ERROR: Exception | None = None
except Exception:
    try:
        # mistralai>=2 exposes the SDK entrypoint from mistralai.client.
        from mistralai.client import Mistral
        _MISTRAL_IMPORT_ERROR = None
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


_DASHSCOPE_SUBMIT_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
_DASHSCOPE_TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks"
_QWEN3ASR_POLL_INITIAL_DELAY = 3.0
_QWEN3ASR_POLL_MAX_DELAY = 10.0
_QWEN3ASR_POLL_TIMEOUT = 7200


class Qwen3ASRProvider(TranscriptionProvider):
    def __init__(self, settings: Qwen3ASRProviderSettings, http_client: Optional[HttpClient] = None) -> None:
        self.settings = settings
        self.http_client = http_client or HttpClient(timeout_seconds=120)

    def transcribe(
        self,
        request: TranscriptionRequest,
        progress_cb: Optional[Callable[[str], None]],
        cancel_event: Event,
    ) -> TranscriptionResult:
        if cancel_event.is_set():
            raise RuntimeError("转写前已取消")

        if progress_cb:
            progress_cb("正在提交 Qwen3 ASR 任务")

        audio_path = request.audio_path
        file_url = str(audio_path.resolve().as_uri())

        task_id = self._submit_task(file_url, request, cancel_event)

        if progress_cb:
            progress_cb("Qwen3 ASR 任务已提交，等待识别完成")
        result_url = self._poll_task(task_id, progress_cb, cancel_event)

        if cancel_event.is_set():
            raise RuntimeError("转写已取消")

        if progress_cb:
            progress_cb("正在下载 Qwen3 ASR 识别结果")
        payload = self._download_result(result_url)

        segments = self._extract_segments(payload)
        text = self._extract_text(payload)
        language = detect_language_code(payload)

        return TranscriptionResult(
            text=text,
            segments=segments,
            language=language,
            raw_payload=payload,
        )

    def _submit_task(
        self,
        file_url: str,
        request: TranscriptionRequest,
        cancel_event: Event,
    ) -> str:
        body: Dict[str, Any] = {
            "model": self.settings.model,
            "input": {"file_url": file_url},
            "parameters": {},
        }
        if request.language_mode == "manual" and request.language:
            body["parameters"]["language_hints"] = [request.language]

        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }

        response = self.http_client.post_json(_DASHSCOPE_SUBMIT_URL, body, headers)
        if response.status_code >= 400:
            raise RuntimeError(
                f"Qwen3 ASR 提交任务失败: HTTP {response.status_code} {response.text[:240]}"
            )
        payload = response.payload if isinstance(response.payload, dict) else {}
        task_id = payload.get("output", {}).get("task_id", "")
        if not task_id:
            raise RuntimeError(f"Qwen3 ASR 未返回 task_id: {response.text[:240]}")
        return task_id

    def _poll_task(
        self,
        task_id: str,
        progress_cb: Optional[Callable[[str], None]],
        cancel_event: Event,
    ) -> str:
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        url = f"{_DASHSCOPE_TASK_URL}/{task_id}"

        delay = _QWEN3ASR_POLL_INITIAL_DELAY
        elapsed = 0.0
        while elapsed < _QWEN3ASR_POLL_TIMEOUT:
            if cancel_event.is_set():
                raise RuntimeError("转写已取消")

            time.sleep(delay)
            elapsed += delay
            delay = min(delay * 1.5, _QWEN3ASR_POLL_MAX_DELAY)

            response = self.http_client.get_json(url, headers)
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Qwen3 ASR 查询任务失败: HTTP {response.status_code} {response.text[:240]}"
                )
            payload = response.payload if isinstance(response.payload, dict) else {}
            status = payload.get("output", {}).get("task_status", "")

            if status == "SUCCEEDED":
                result_url = payload.get("output", {}).get("result", {}).get("transcription_url", "")
                if not result_url:
                    raise RuntimeError(f"Qwen3 ASR 任务成功但未返回结果 URL: {response.text[:240]}")
                return result_url
            if status in ("FAILED", "CANCELED", "UNKNOWN"):
                code = payload.get("output", {}).get("code", "")
                message = payload.get("output", {}).get("message", "")
                raise RuntimeError(f"Qwen3 ASR 任务失败: {status} {code} {message}")

            if progress_cb:
                progress_cb(f"Qwen3 ASR 识别中... ({int(elapsed)}s)")

        raise RuntimeError(f"Qwen3 ASR 任务超时（{_QWEN3ASR_POLL_TIMEOUT}s），task_id={task_id}")

    def _download_result(self, result_url: str) -> Dict[str, Any]:
        response = self.http_client.get_json(result_url)
        if response.status_code >= 400:
            raise RuntimeError(
                f"Qwen3 ASR 下载结果失败: HTTP {response.status_code} {response.text[:240]}"
            )
        if isinstance(response.payload, dict):
            return response.payload
        try:
            return json.loads(response.text)
        except Exception:
            return {"text": response.text}

    @staticmethod
    def _extract_segments(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
        segments: list[Dict[str, Any]] = []
        for transcript in payload.get("transcripts", []):
            for sentence in transcript.get("sentences", []):
                begin_ms = sentence.get("begin_time", 0)
                end_ms = sentence.get("end_time", 0)
                text = sentence.get("text", "")
                if not text:
                    continue
                seg: Dict[str, Any] = {
                    "start": begin_ms / 1000.0,
                    "end": end_ms / 1000.0,
                    "text": text,
                }
                if "speaker_id" in sentence:
                    seg["speaker"] = sentence["speaker_id"]
                if "emotion" in sentence:
                    seg["emotion"] = sentence["emotion"]
                if "language" in sentence:
                    seg["language"] = sentence["language"]
                segments.append(seg)
        return segments

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> str:
        parts: list[str] = []
        for transcript in payload.get("transcripts", []):
            for sentence in transcript.get("sentences", []):
                text = sentence.get("text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)


def summarize_empty_transcription_response(payload: Dict[str, Any], raw_text: str) -> str:
    if payload:
        keys = ", ".join(sorted(str(key) for key in payload.keys())[:8])
        if keys:
            return f" 响应字段: {keys}。"
    snippet = raw_text.strip().replace("\r", " ").replace("\n", " ")
    if snippet:
        return f" 原始响应片段: {snippet[:240]}"
    return " 请检查服务端日志，以及所选模型是否真的支持 OpenAI 兼容的 audio/transcriptions 输出。"
