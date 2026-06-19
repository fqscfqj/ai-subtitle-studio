from __future__ import annotations

import unittest

from subtitle_studio.http_client import HttpResponse
from subtitle_studio.providers.translation import OpenAICompatibleChatBackend


class FakeHttpClient:
    def __init__(self) -> None:
        self.calls = []

    def post_json(self, url, payload, headers):
        self.calls.append({"url": url, "payload": dict(payload), "headers": dict(headers)})
        return HttpResponse(
            status_code=200,
            payload={"choices": [{"message": {"content": "ok"}}]},
            text="",
        )


class ThinkingToggleTests(unittest.TestCase):
    def test_openai_compatible_thinking_is_sent_as_top_level_field(self) -> None:
        fake_client = FakeHttpClient()
        backend = OpenAICompatibleChatBackend(
            base_url="https://example.com/v1",
            api_key="secret",
            http_client=fake_client,
        )

        result = backend.complete(
            model="deepseek-v4-flash",
            system_prompt="sys",
            user_prompt="user",
            thinking_enabled=True,
            reasoning_effort="high",
        )

        self.assertEqual(result, "ok")
        self.assertEqual(len(fake_client.calls), 1)
        payload = fake_client.calls[0]["payload"]
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertNotIn("extra_body", payload)
        self.assertNotIn("temperature", payload)

    def test_openai_compatible_thinking_disabled_is_explicitly_sent(self) -> None:
        fake_client = FakeHttpClient()
        backend = OpenAICompatibleChatBackend(
            base_url="https://example.com/v1",
            api_key="secret",
            http_client=fake_client,
        )

        result = backend.complete(
            model="deepseek-v4-flash",
            system_prompt="sys",
            user_prompt="user",
            thinking_enabled=False,
            reasoning_effort="high",
        )

        self.assertEqual(result, "ok")
        self.assertEqual(len(fake_client.calls), 1)
        payload = fake_client.calls[0]["payload"]
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["temperature"], 0.2)
        self.assertNotIn("reasoning_effort", payload)

    def test_openai_compatible_custom_temperature_is_forwarded_when_thinking_disabled(self) -> None:
        fake_client = FakeHttpClient()
        backend = OpenAICompatibleChatBackend(
            base_url="https://example.com/v1",
            api_key="secret",
            http_client=fake_client,
        )

        backend.complete(
            model="deepseek-v4-flash",
            system_prompt="sys",
            user_prompt="user",
            thinking_enabled=False,
            reasoning_effort="high",
            temperature=0.7,
        )

        payload = fake_client.calls[0]["payload"]
        self.assertEqual(payload["temperature"], 0.7)


if __name__ == "__main__":
    unittest.main()
