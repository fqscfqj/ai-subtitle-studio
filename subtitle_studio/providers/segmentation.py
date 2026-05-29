from __future__ import annotations

import json
from threading import Event
from typing import Any, Dict, List, Optional

from ..http_client import HttpClient
from ..models import (
    SegmentationProvider,
    SegmentationRequest,
    SegmentationSettings,
    TaskCancelled,
    WordToken,
)
from ..utils import extract_chat_text
from .translation import normalize_chat_completions_url


class OpenAICompatibleSegmentationBackend:
    def __init__(self, base_url: str, api_key: str, http_client: Optional[HttpClient] = None) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.http_client = http_client or HttpClient()

    def complete(self, model: str, system_prompt: str, user_prompt: str) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        response = self.http_client.post_json(
            normalize_chat_completions_url(self.base_url),
            payload=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"智能分段接口返回错误: HTTP {response.status_code} {response.text[:240]}"
            )
        payload = response.payload if isinstance(response.payload, dict) else {}
        content = extract_chat_text(payload)
        if not content:
            raise RuntimeError("智能分段接口未返回可用文本")
        return content


class ChatCompletionSegmentationProvider(SegmentationProvider):
    def __init__(
        self,
        backend: OpenAICompatibleSegmentationBackend,
        max_attempts: int = 3,
        max_words_per_window: int = 180,
    ) -> None:
        self.backend = backend
        self.max_attempts = max(1, max_attempts)
        self.max_words_per_window = max(20, max_words_per_window)

    def segment_words(
        self,
        words: list[WordToken],
        request: SegmentationRequest,
        cancel_event: Event,
    ) -> list[tuple[int, int]]:
        normalized_words = self._normalize_words(words)
        if not normalized_words:
            return []

        source_language = request.source_language.strip() if request.source_language else ""
        language_hint = (
            f"输入语言大致为 `{source_language}`。"
            if source_language and source_language != "und"
            else "输入语言未知，请结合词序、标点和停顿自行判断。"
        )
        system_prompt = (
            "你是专业字幕智能分段器。"
            "你将收到按顺序排列的词级时间戳 JSON 数组。"
            "请只根据语义、标点和停顿，把连续词切成适合字幕阅读的自然短句。"
            "严禁改写、删词、重排，也不要输出任何解释。"
            "只返回 JSON 数组；每个元素包含 start_index 和 end_index，且为闭区间。"
            "返回结果必须完整覆盖当前窗口内全部词索引，不能缺失、不能重叠、不能越界。"
            "优先在句号、问号、感叹号、分号、明显停顿处断句；避免过长片段。"
            f"{language_hint}"
        )

        merged_ranges: list[tuple[int, int]] = []
        for start_index, end_index in self._build_windows(normalized_words):
            if cancel_event.is_set():
                raise TaskCancelled("智能分段前已取消")

            window_words = normalized_words[start_index:end_index]
            local_ranges = self._segment_window(
                window_words=window_words,
                request=request,
                system_prompt=system_prompt,
                cancel_event=cancel_event,
            )
            merged_ranges.extend(
                (local_start + start_index, local_end + start_index)
                for local_start, local_end in local_ranges
            )

        _validate_segmentation_ranges(merged_ranges, len(normalized_words))
        return merged_ranges

    def _segment_window(
        self,
        window_words: List[Dict[str, Any]],
        request: SegmentationRequest,
        system_prompt: str,
        cancel_event: Event,
    ) -> list[tuple[int, int]]:
        last_content = ""
        last_error = ""
        user_prompt = self._build_user_prompt(window_words)

        for attempt in range(1, self.max_attempts + 1):
            if cancel_event.is_set():
                raise TaskCancelled("智能分段前已取消")

            content = self.backend.complete(request.model, system_prompt, user_prompt)
            last_content = content
            if cancel_event.is_set():
                raise TaskCancelled("智能分段前已取消")
            try:
                parsed = _parse_segmentation_ranges(content)
                _validate_segmentation_ranges(parsed, len(window_words))
                return parsed
            except Exception as exc:
                last_error = str(exc).strip() or "未知错误"
                if attempt >= self.max_attempts:
                    preview = last_content.strip().replace("\n", " ")
                    if len(preview) > 220:
                        preview = preview[:220] + "..."
                    raise RuntimeError(
                        f"智能分段失败：{last_error} | 返回片段：{preview}"
                    )
        raise RuntimeError("智能分段失败：未获得有效结果")

    def _build_user_prompt(self, window_words: List[Dict[str, Any]]) -> str:
        payload = [
            {
                "index": idx,
                "start": round(float(word["start"]), 3),
                "end": round(float(word["end"]), 3),
                "text": str(word["text"]),
            }
            for idx, word in enumerate(window_words)
        ]
        return (
            "请为下面这组词级时间戳做字幕智能分段。"
            "输出格式示例："
            "[{\"start_index\":0,\"end_index\":5},{\"start_index\":6,\"end_index\":11}]。"
            "不要输出 Markdown，不要输出额外字段。"
            f"\n词序列：{json.dumps(payload, ensure_ascii=False)}"
        )

    def _normalize_words(self, words: list[WordToken]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for word in words:
            try:
                start = float(word["start"])
                end = float(word["end"])
            except Exception:
                continue
            text = str(word.get("text", ""))
            if not text or not text.strip():
                continue
            normalized.append({"start": start, "end": max(start, end), "text": text})
        return normalized

    def _build_windows(self, words: List[Dict[str, Any]]) -> list[tuple[int, int]]:
        if len(words) <= self.max_words_per_window:
            return [(0, len(words))]

        windows: list[tuple[int, int]] = []
        start = 0
        while start < len(words):
            hard_end = min(len(words), start + self.max_words_per_window)
            if hard_end >= len(words):
                windows.append((start, len(words)))
                break

            soft_end = self._find_soft_break(words, start, hard_end)
            if soft_end <= start:
                soft_end = hard_end
            windows.append((start, soft_end))
            start = soft_end
        return windows

    def _find_soft_break(self, words: List[Dict[str, Any]], start: int, hard_end: int) -> int:
        search_start = max(start + 1, hard_end - 36)
        for index in range(hard_end - 1, search_start - 1, -1):
            token_text = str(words[index].get("text", ""))
            has_sentence_boundary = any(ch in token_text for ch in ".!?。！？；;")
            pause_after = 0.0
            if index + 1 < len(words):
                pause_after = max(0.0, float(words[index + 1]["start"]) - float(words[index]["end"]))
            if has_sentence_boundary or pause_after >= 0.6:
                return index + 1
        return hard_end


def build_segmentation_provider(settings: SegmentationSettings) -> SegmentationProvider | None:
    if not settings.enabled:
        return None
    return ChatCompletionSegmentationProvider(
        OpenAICompatibleSegmentationBackend(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
    )


def _parse_segmentation_ranges(raw_text: str) -> list[tuple[int, int]]:
    parsed = _load_json_candidate(raw_text)
    if isinstance(parsed, dict):
        for key in ("segments", "ranges", "items", "data", "output"):
            candidate = parsed.get(key)
            if isinstance(candidate, list):
                parsed = candidate
                break
    if not isinstance(parsed, list):
        raise RuntimeError("智能分段结果不是 JSON 数组")

    ranges: list[tuple[int, int]] = []
    for item in parsed:
        if isinstance(item, list) and len(item) >= 2:
            start = _coerce_index(item[0])
            end = _coerce_index(item[1])
        elif isinstance(item, dict):
            start = _coerce_index(
                item.get("start_index", item.get("start", item.get("from", item.get("begin"))))
            )
            end = _coerce_index(
                item.get("end_index", item.get("end", item.get("to", item.get("stop"))))
            )
            if (start is None or end is None) and isinstance(item.get("indices"), list) and len(item["indices"]) >= 2:
                start = _coerce_index(item["indices"][0])
                end = _coerce_index(item["indices"][1])
        else:
            start = None
            end = None
        if start is None or end is None:
            raise RuntimeError("智能分段结果包含非法索引")
        ranges.append((start, end))
    return ranges


def _coerce_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            number = float(token)
        except Exception:
            return None
        if not number.is_integer():
            return None
        return int(number)
    return None


def _validate_segmentation_ranges(ranges: list[tuple[int, int]], word_count: int) -> None:
    if word_count <= 0:
        if ranges:
            raise RuntimeError("空词序列不应返回分段结果")
        return
    if not ranges:
        raise RuntimeError("智能分段结果为空")

    expected_start = 0
    for start, end in ranges:
        if start < 0 or end < 0:
            raise RuntimeError("智能分段索引不能为负数")
        if start > end:
            raise RuntimeError("智能分段起止索引非法")
        if start != expected_start:
            raise RuntimeError("智能分段结果未完整覆盖全部词索引")
        expected_start = end + 1
    if expected_start != word_count:
        raise RuntimeError("智能分段结果未完整覆盖全部词索引")


def _load_json_candidate(raw_text: str) -> Any:
    text = raw_text.replace("\ufeff", "").strip()
    candidates: List[str] = []
    base = _strip_code_fence(text)
    if base:
        candidates.append(base)
    for open_char, close_char in (("[", "]"), ("{", "}")):
        snippet = _find_balanced_json(base, open_char, close_char)
        if snippet:
            candidates.append(snippet)

    seen = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            return json.loads(normalized)
        except Exception:
            continue
    raise RuntimeError("智能分段结果不是有效 JSON")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _find_balanced_json(text: str, open_char: str, close_char: str) -> str:
    in_string = False
    escaped = False
    depth = 0
    start = -1
    for index, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            if depth == 0:
                start = index
            depth += 1
            continue
        if ch == close_char and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : index + 1]
    return ""
