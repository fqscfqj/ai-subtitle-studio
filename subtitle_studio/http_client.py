from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import requests

from .constants import MAX_RETRIES, RETRYABLE_EXCEPTION_TYPES, RETRYABLE_STATUS_CODES, RETRY_BASE_DELAY

_log = logging.getLogger(__name__)


@dataclass
class HttpResponse:
    status_code: int
    payload: Any
    text: str


class HttpClient:
    def __init__(self, timeout_seconds: int = 180) -> None:
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Optional[Mapping[str, str]] = None,
    ) -> HttpResponse:
        def _do() -> requests.Response:
            return self.session.post(
                url,
                json=dict(payload),
                headers=dict(headers or {}),
                timeout=self.timeout_seconds,
            )
        return self._build_response(self._request_with_retry(_do, url))

    def get_json(
        self,
        url: str,
        headers: Optional[Mapping[str, str]] = None,
    ) -> HttpResponse:
        def _do() -> requests.Response:
            return self.session.get(
                url,
                headers=dict(headers or {}),
                timeout=self.timeout_seconds,
            )
        return self._build_response(self._request_with_retry(_do, url))

    def post_multipart(
        self,
        url: str,
        data: Mapping[str, Any],
        files: Mapping[str, tuple[str, bytes, str]],
        headers: Optional[Mapping[str, str]] = None,
    ) -> HttpResponse:
        def _do() -> requests.Response:
            return self.session.post(
                url,
                data=dict(data),
                files=dict(files),
                headers=dict(headers or {}),
                timeout=self.timeout_seconds,
            )
        return self._build_response(self._request_with_retry(_do, url))

    def _request_with_retry(self, do_request, url: str) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = do_request()
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    return response
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    _log.warning(
                        "HTTP %d from %s，%d 秒后重试 (%d/%d)",
                        response.status_code, url, delay, attempt + 1, MAX_RETRIES,
                    )
                    time.sleep(delay)
                    continue
                return response
            except RETRYABLE_EXCEPTION_TYPES as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    _log.warning(
                        "请求 %s 失败 (%s)，%d 秒后重试 (%d/%d)",
                        url, exc, delay, attempt + 1, MAX_RETRIES,
                    )
                    time.sleep(delay)
                    continue
                raise
        # Should not reach here, but just in case
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"HTTP 请求 {url} 失败，已重试 {MAX_RETRIES} 次")

    def _build_response(self, response: requests.Response) -> HttpResponse:
        raw_text = response.text
        try:
            payload = response.json()
        except Exception:
            payload = None
        return HttpResponse(status_code=response.status_code, payload=payload, text=raw_text)

