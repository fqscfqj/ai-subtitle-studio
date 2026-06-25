from __future__ import annotations

import unittest

from subtitle_studio.utils import extract_segments, sanitize_transcribed_text


class SanitizeTranscribedTextTests(unittest.TestCase):
    def test_filters_stage_directions(self) -> None:
        for raw in ("*Sigh*", "[Music]", "(applause)", "（叹气）"):
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_transcribed_text(raw), "")

    def test_filters_number_and_symbol_fragments(self) -> None:
        for raw in ("-", "- 10.", "...", "• 3)"):
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_transcribed_text(raw), "")

    def test_keeps_pure_number_utterances(self) -> None:
        for raw in ("10", "10.", "2025"):
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_transcribed_text(raw), raw)

    def test_filters_common_filler_utterances(self) -> None:
        for raw in ("Mm-hmm.", "Uh-huh.", "Hmm...", "Um..."):
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_transcribed_text(raw), "")

    def test_filters_hallucination_artifacts(self) -> None:
        for raw in ("parakeet Й", "parakeet П", "长尾鹦鹉 Й", "长尾鹦鹉 П", "长尾鹦鹉 Й P", "parakeet Й P"):
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_transcribed_text(raw), "")

    def test_keeps_parakeet_in_normal_context(self) -> None:
        self.assertEqual(sanitize_transcribed_text("I saw a parakeet today."), "I saw a parakeet today.")

    def test_normalizes_dialogue_dash(self) -> None:
        self.assertEqual(sanitize_transcribed_text("- Yeah, it's the first time."), "Yeah, it's the first time.")
        self.assertEqual(sanitize_transcribed_text("- Meili."), "Meili.")

    def test_keeps_real_short_utterances(self) -> None:
        cases = {
            "Yes.": "Yes.",
            "No.": "No.",
            "Okay.": "Okay.",
            "I'm here.": "I'm here.",
            "好的。": "好的。",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_transcribed_text(raw), expected)

    def test_extract_segments_skips_filtered_noise(self) -> None:
        segments = extract_segments(
            {
                "segments": [
                    {"start": 0.0, "end": 0.5, "text": "*Sigh*"},
                    {"start": 0.5, "end": 1.0, "text": "Mm-hmm."},
                    {"start": 1.0, "end": 2.0, "text": "I’m here."},
                ]
            }
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "I’m here.")
        self.assertAlmostEqual(segments[0]["start"], 1.0, places=3)
        self.assertAlmostEqual(segments[0]["end"], 2.0, places=3)

    def test_extract_segments_builds_synthetic_segment_from_top_level_words(self) -> None:
        segments = extract_segments(
            {
                "text": "hello world",
                "words": [
                    {"start": 0.0, "end": 0.5, "word": "hello"},
                    {"start": 0.5, "end": 1.0, "word": "world"},
                ],
            }
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "hello world")
        self.assertEqual(len(segments[0]["words"]), 2)
        self.assertAlmostEqual(segments[0]["start"], 0.0, places=3)
        self.assertAlmostEqual(segments[0]["end"], 1.0, places=3)

    def test_extract_segments_attaches_top_level_words_to_segments(self) -> None:
        segments = extract_segments(
            {
                "segments": [
                    {"start": 0.0, "end": 0.5, "text": "hello"},
                    {"start": 0.5, "end": 1.0, "text": "world"},
                ],
                "words": [
                    {"start": 0.0, "end": 0.4, "word": "hello"},
                    {"start": 0.5, "end": 1.0, "word": "world"},
                ],
            }
        )

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["words"][0]["text"], "hello")
        self.assertEqual(segments[1]["words"][0]["text"], "world")

    def test_extract_segments_normalizes_word_level_nanosecond_timestamps(self) -> None:
        segments = extract_segments(
            {
                "text": "hello world",
                "words": [
                    {"start": 152000000.0, "end": 1000000152.0, "word": "hello"},
                    {"start": 1000000152.0, "end": 2152000000.0, "word": "world"},
                ],
            }
        )

        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0]["words"][0]["start"], 0.152, places=3)
        self.assertAlmostEqual(segments[0]["words"][0]["end"], 1.0, places=3)
        self.assertAlmostEqual(segments[0]["words"][1]["start"], 1.0, places=3)
        self.assertAlmostEqual(segments[0]["words"][1]["end"], 2.152, places=3)
        self.assertAlmostEqual(segments[0]["start"], 0.152, places=3)
        self.assertAlmostEqual(segments[0]["end"], 2.152, places=3)

    def test_extract_segments_handles_nemo_start_time_end_time(self) -> None:
        """NeMo / parakeet 后端返回 start_time/end_time（微秒整数）。"""
        segments = extract_segments(
            {
                "text": "hello world",
                "words": [
                    {"start_time": 152000, "end_time": 500000, "word": "hello"},
                    {"start_time": 500000, "end_time": 1000000, "word": "world"},
                ],
            }
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0]["words"]), 2)
        self.assertAlmostEqual(segments[0]["words"][0]["start"], 0.152, places=3)
        self.assertAlmostEqual(segments[0]["words"][0]["end"], 0.5, places=3)
        self.assertAlmostEqual(segments[0]["words"][1]["start"], 0.5, places=3)
        self.assertAlmostEqual(segments[0]["words"][1]["end"], 1.0, places=3)

    def test_extract_segments_normalizes_microsecond_timestamps(self) -> None:
        """start_time/end_time 微秒值通过 _extract_word_tokens 转换为秒。"""
        segments = extract_segments(
            {
                "text": "hello world",
                "words": [
                    {"start_time": 152000, "end_time": 500000, "word": "hello"},
                    {"start_time": 500000, "end_time": 1000000, "word": "world"},
                ],
            }
        )
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0]["words"][0]["start"], 0.152, places=3)
        self.assertAlmostEqual(segments[0]["words"][0]["end"], 0.5, places=3)
        self.assertAlmostEqual(segments[0]["words"][1]["start"], 0.5, places=3)
        self.assertAlmostEqual(segments[0]["words"][1]["end"], 1.0, places=3)

    def test_extract_segments_synthesizes_words_from_text_only_tokens(self) -> None:
        """tokens 只有文本无时间戳时，从 segment 时间均匀插值生成 words。"""
        segments = extract_segments(
            {
                "text": "Hello world",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "Hello world",
                        "tokens": [
                            {"token": "Hello"},
                            {"token": "world"},
                        ],
                    }
                ],
            }
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0]["words"]), 2)
        self.assertAlmostEqual(segments[0]["words"][0]["start"], 0.0, places=3)
        self.assertAlmostEqual(segments[0]["words"][0]["end"], 0.5, places=3)
        self.assertEqual(segments[0]["words"][0]["text"], "Hello")
        self.assertAlmostEqual(segments[0]["words"][1]["start"], 0.5, places=3)
        self.assertAlmostEqual(segments[0]["words"][1]["end"], 1.0, places=3)
        self.assertEqual(segments[0]["words"][1]["text"], "world")

    def test_extract_segments_synthesizes_words_from_string_tokens(self) -> None:
        """tokens 为纯字符串列表时也能合成 words。"""
        segments = extract_segments(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 0.6,
                        "text": "Hi there",
                        "tokens": ["Hi", "there"],
                    }
                ],
            }
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0]["words"]), 2)
        self.assertAlmostEqual(segments[0]["words"][0]["start"], 0.0, places=3)
        self.assertAlmostEqual(segments[0]["words"][0]["end"], 0.3, places=3)
        self.assertEqual(segments[0]["words"][0]["text"], "Hi")
        self.assertAlmostEqual(segments[0]["words"][1]["start"], 0.3, places=3)
        self.assertAlmostEqual(segments[0]["words"][1]["end"], 0.6, places=3)
        self.assertEqual(segments[0]["words"][1]["text"], "there")

    def test_extract_segments_synthesizes_words_from_text_when_tokens_are_ids(self) -> None:
        """tokens 为 Whisper token ID（整数/None）时，回退到从 segment text 拆词。"""
        segments = extract_segments(
            {
                "text": "Hello world",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "Hello world",
                        "tokens": [50364, 15339, 1002, 50464],
                    }
                ],
            }
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(len(segments[0]["words"]), 2)
        self.assertAlmostEqual(segments[0]["words"][0]["start"], 0.0, places=3)
        self.assertAlmostEqual(segments[0]["words"][0]["end"], 0.5, places=3)
        self.assertEqual(segments[0]["words"][0]["text"], "Hello")
        self.assertEqual(segments[0]["words"][1]["text"], "world")


if __name__ == "__main__":
    unittest.main()
