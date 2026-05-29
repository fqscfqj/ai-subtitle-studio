from __future__ import annotations

import unittest
from threading import Event

from subtitle_studio.http_client import HttpResponse
from subtitle_studio.models import SegmentationRequest, SegmentationSettings, TaskCancelled
from subtitle_studio.providers.segmentation import (
    ChatCompletionSegmentationProvider,
    OpenAICompatibleSegmentationBackend,
    build_segmentation_provider,
)


class FakeHttpClient:
    def __init__(self, response_text: str = '[{"start_index":0,"end_index":1}]') -> None:
        self.response_text = response_text
        self.calls = []

    def post_json(self, url, payload, headers):
        self.calls.append({"url": url, "payload": dict(payload), "headers": dict(headers)})
        return HttpResponse(
            status_code=200,
            payload={"choices": [{"message": {"content": self.response_text}}]},
            text="",
        )


class SegmentationProviderTests(unittest.TestCase):
    def test_openai_compatible_backend_uses_chat_completions(self) -> None:
        fake_client = FakeHttpClient('[{"start_index":0,"end_index":0}]')
        backend = OpenAICompatibleSegmentationBackend(
            base_url="https://segment.example.com/v1",
            api_key="secret",
            http_client=fake_client,
        )

        content = backend.complete(
            model="segment-model",
            system_prompt="sys",
            user_prompt="user",
        )

        self.assertEqual(content, '[{"start_index":0,"end_index":0}]')
        self.assertEqual(len(fake_client.calls), 1)
        call = fake_client.calls[0]
        self.assertEqual(call["url"], "https://segment.example.com/v1/chat/completions")
        self.assertEqual(call["payload"]["model"], "segment-model")
        self.assertEqual(call["payload"]["temperature"], 0.1)
        self.assertEqual(call["headers"]["Authorization"], "Bearer secret")

    def test_segment_words_returns_valid_ranges(self) -> None:
        fake_client = FakeHttpClient('[{"start_index":0,"end_index":1},{"start_index":2,"end_index":2}]')
        provider = ChatCompletionSegmentationProvider(
            OpenAICompatibleSegmentationBackend(
                base_url="https://segment.example.com/v1",
                api_key="secret",
                http_client=fake_client,
            )
        )

        ranges = provider.segment_words(
            words=[
                {"start": 0.0, "end": 0.2, "text": "hello"},
                {"start": 0.2, "end": 0.5, "text": "world"},
                {"start": 0.5, "end": 0.9, "text": "!"},
            ],
            request=SegmentationRequest(model="segment-model", source_language="en"),
            cancel_event=Event(),
        )

        self.assertEqual(ranges, [(0, 1), (2, 2)])
        payload = fake_client.calls[0]["payload"]
        self.assertEqual(payload["model"], "segment-model")
        self.assertIn("词序列", payload["messages"][1]["content"])

    def test_segment_words_rejects_gapped_ranges(self) -> None:
        fake_client = FakeHttpClient('[{"start_index":0,"end_index":0},{"start_index":2,"end_index":2}]')
        provider = ChatCompletionSegmentationProvider(
            OpenAICompatibleSegmentationBackend(
                base_url="https://segment.example.com/v1",
                api_key="secret",
                http_client=fake_client,
            ),
            max_attempts=1,
        )

        with self.assertRaises(RuntimeError) as ctx:
            provider.segment_words(
                words=[
                    {"start": 0.0, "end": 0.2, "text": "hello"},
                    {"start": 0.2, "end": 0.5, "text": "world"},
                    {"start": 0.5, "end": 0.9, "text": "again"},
                ],
                request=SegmentationRequest(model="segment-model", source_language="en"),
                cancel_event=Event(),
            )

        self.assertIn("未完整覆盖", str(ctx.exception))

    def test_segment_words_honors_cancel_event(self) -> None:
        fake_client = FakeHttpClient()
        provider = ChatCompletionSegmentationProvider(
            OpenAICompatibleSegmentationBackend(
                base_url="https://segment.example.com/v1",
                api_key="secret",
                http_client=fake_client,
            )
        )
        cancel_event = Event()
        cancel_event.set()

        with self.assertRaises(TaskCancelled):
            provider.segment_words(
                words=[{"start": 0.0, "end": 0.2, "text": "hello"}],
                request=SegmentationRequest(model="segment-model", source_language="en"),
                cancel_event=cancel_event,
            )

    def test_build_segmentation_provider_returns_none_when_disabled(self) -> None:
        settings = SegmentationSettings(enabled=False)
        self.assertIsNone(build_segmentation_provider(settings))


if __name__ == "__main__":
    unittest.main()
