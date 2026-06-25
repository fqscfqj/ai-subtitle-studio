from __future__ import annotations

import logging
from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict, Optional

_log = logging.getLogger(__name__)

from ..http_client import HttpClient
from ..models import (
    TranscriptionProvider,
    TranscriptionRequest,
    TranscriptionResult,
    WhisperProviderSettings,
)
from ..utils import detect_language_code, extract_segments, extract_text


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
                _log.warning("服务端不支持 timestamp_granularities，回退为无时间戳模式。原始错误: %s", response.text[:200])
                fallback_data = self._build_form_data(request, include_timestamps=False)
                response = self.http_client.post_multipart(endpoint, data=fallback_data, files=files, headers=headers)

        if response.status_code >= 400:
            raise RuntimeError(
                f"Whisper 兼容接口返回错误: HTTP {response.status_code} {response.text[:240]}"
            )
        payload = response.payload if isinstance(response.payload, dict) else {"text": response.text}

        segments = extract_segments(payload)
        if request.timestamp_granularity == "word" and segments:
            has_words = any(isinstance(s.get("words"), list) and s["words"] for s in segments)
            if not has_words:
                seg0 = payload["segments"][0] if payload.get("segments") and isinstance(payload["segments"][0], dict) else {}
                tokens_val = seg0.get("tokens")
                token_sample = None
                if isinstance(tokens_val, list) and tokens_val:
                    token_sample = tokens_val[0]
                _log.warning(
                    "请求了 word 级时间戳但响应未提取到 words。"
                    "响应顶层键: %s；segments[0] 键: %s；tokens[0] 样例: %s",
                    sorted(payload.keys()) if isinstance(payload, dict) else "N/A",
                    sorted(seg0.keys()) if seg0 else "N/A",
                    token_sample,
                )
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
            ".opus": "audio/opus",
            ".ogg": "audio/ogg",
        }.get(suffix, "application/octet-stream")




def summarize_empty_transcription_response(payload: Dict[str, Any], raw_text: str) -> str:
    if payload:
        keys = ", ".join(sorted(str(key) for key in payload.keys())[:8])
        if keys:
            return f" 响应字段: {keys}。"
    snippet = raw_text.strip().replace("\r", " ").replace("\n", " ")
    if snippet:
        return f" 原始响应片段: {snippet[:240]}"
    return " 请检查服务端日志，以及所选模型是否真的支持 OpenAI 兼容的 audio/transcriptions 输出。"
