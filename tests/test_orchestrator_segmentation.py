from __future__ import annotations

import unittest
from pathlib import Path
from threading import Event

from subtitle_studio.media import AudioChunk
from subtitle_studio.models import AppSettings, TranscriptionResult
from subtitle_studio.orchestrator import TaskRunner


class FakeSequenceProvider:
    def __init__(self, results) -> None:
        self.results = list(results)
        self.calls = 0

    def transcribe(self, request, progress_cb, cancel_event):
        result = self.results[self.calls]
        self.calls += 1
        return result


class FakeSegmentationProvider:
    def __init__(self, ranges) -> None:
        self.ranges = list(ranges)
        self.calls = []

    def segment_words(self, words, request, cancel_event):
        self.calls.append({"words": [dict(word) for word in words], "request": request})
        return list(self.ranges)


class FakeTranslationProvider:
    def __init__(self) -> None:
        self.calls = []

    def translate_lines(self, lines, request, cancel_event, parallel_workers=1):
        self.calls.append(list(lines))
        return [f"tr:{line}" for line in lines]


class SegmentationAwareTaskRunner(TaskRunner):
    def __init__(self, settings, segmentation_provider=None, translation_provider=None) -> None:
        super().__init__(settings)
        self._segmentation_provider = segmentation_provider
        self._translation_provider = translation_provider

    def _build_segmentation_provider(self):
        return self._segmentation_provider

    def _build_translation_provider(self):
        return self._translation_provider


class OrchestratorSegmentationTests(unittest.TestCase):
    def test_run_transcription_chunks_offsets_nested_word_timestamps(self) -> None:
        runner = TaskRunner(AppSettings())
        provider = FakeSequenceProvider(
            [
                TranscriptionResult(text="", segments=[], language="", raw_payload={"text": "", "segments": []}),
                TranscriptionResult(
                    text="hello world",
                    segments=[
                        {
                            "start": 0.0,
                            "end": 1.0,
                            "text": "hello world",
                            "words": [
                                {"start": 0.0, "end": 0.4, "text": "hello"},
                                {"start": 0.5, "end": 1.0, "text": "world"},
                            ],
                        }
                    ],
                    language="en",
                    raw_payload={"text": "hello world", "segments": []},
                ),
            ]
        )
        chunks = [
            AudioChunk(path=Path("chunk1.wav"), start_offset=0.0, end_offset=1.0),
            AudioChunk(path=Path("chunk2.wav"), start_offset=5.0, end_offset=6.0),
        ]

        result = runner._run_transcription_chunks(
            source_path=Path("input.wav"),
            provider=provider,
            chunks=chunks,
            report=lambda stage, progress, message: None,
            cancel_event=Event(),
        )

        self.assertEqual(len(result.segments), 1)
        self.assertAlmostEqual(result.segments[0]["start"], 5.0, places=3)
        self.assertAlmostEqual(result.segments[0]["end"], 6.0, places=3)
        self.assertAlmostEqual(result.segments[0]["words"][0]["start"], 5.0, places=3)
        self.assertAlmostEqual(result.segments[0]["words"][0]["end"], 5.4, places=3)
        self.assertAlmostEqual(result.segments[0]["words"][1]["start"], 5.5, places=3)
        self.assertAlmostEqual(result.segments[0]["words"][1]["end"], 6.0, places=3)

    def test_intelligent_segmentation_rebuilds_segments_before_translation(self) -> None:
        settings = AppSettings()
        settings.segmentation.enabled = True
        settings.segmentation.model = "segment-model"
        settings.transcription.timestamp_granularity = "word"
        settings.translation.mode = "openai"
        segmentation_provider = FakeSegmentationProvider([(0, 1), (2, 3)])
        translation_provider = FakeTranslationProvider()
        runner = SegmentationAwareTaskRunner(
            settings,
            segmentation_provider=segmentation_provider,
            translation_provider=translation_provider,
        )
        raw_result = TranscriptionResult(
            text="hello brave new world",
            segments=[
                {
                    "start": 0.0,
                    "end": 1.8,
                    "text": "hello brave new world",
                    "words": [
                        {"start": 0.0, "end": 0.3, "text": "hello"},
                        {"start": 0.3, "end": 0.8, "text": "brave"},
                        {"start": 1.0, "end": 1.3, "text": "new"},
                        {"start": 1.3, "end": 1.8, "text": "world"},
                    ],
                }
            ],
            language="en",
            raw_payload={"text": "hello brave new world", "segments": []},
        )

        segmented = runner._apply_intelligent_segmentation(raw_result, Event())
        translated_segments, translated_text = runner._translate_transcription_result(segmented, Event())

        self.assertEqual(len(segmentation_provider.calls), 1)
        self.assertEqual([seg["text"] for seg in segmented.segments], ["hello brave", "new world"])
        self.assertEqual(translation_provider.calls[0], ["hello brave", "new world"])
        self.assertEqual([seg["text"] for seg in translated_segments], ["tr:hello brave", "tr:new world"])
        self.assertEqual(translated_text, "tr:hello brave\ntr:new world")


if __name__ == "__main__":
    unittest.main()
