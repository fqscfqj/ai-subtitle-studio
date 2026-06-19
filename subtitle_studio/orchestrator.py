from __future__ import annotations

import tempfile
from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict, List, Optional

from .config import find_ffmpeg
from .constants import SUBTITLE_EXTENSIONS
from .media import (
    AudioChunk,
    cleanup_paths,
    prepare_audio_source,
    split_audio_with_vad,
)
from .models import (
    AppSettings,
    SegmentationRequest,
    TaskCancelled,
    TranscriptionRequest,
    TranscriptionResult,
    TranslationRequest,
)
from .providers.segmentation import build_segmentation_provider
from .providers.transcription import (
    WhisperOpenAICompatibleProvider,
    summarize_empty_transcription_response,
)
from .providers.translation import build_translation_provider
from .utils import (
    collect_word_tokens,
    detect_language_code,
    extract_subtitle_source,
    infer_language_code_from_filename,
    join_word_tokens,
    normalize_language_code,
    sanitize_transcribed_text,
    trim_language_suffix_from_stem,
)
from .writers import write_transcription_outputs, write_translation_outputs


class TaskRunner:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def run_task(
        self,
        source_path: Path,
        report: Callable[[str, int, str], None],
        cancel_event: Event,
    ) -> Dict[str, str]:
        if source_path.suffix.lower() in SUBTITLE_EXTENSIONS:
            return self.run_subtitle_translation_task(source_path, report, cancel_event)
        return self.run_transcription_task(source_path, report, cancel_event)

    def run_transcription_task(
        self,
        source_path: Path,
        report: Callable[[str, int, str], None],
        cancel_event: Event,
    ) -> Dict[str, str]:
        cleanup: list[Path] = []
        try:
            if cancel_event.is_set():
                raise TaskCancelled("启动前已取消")

            report("Preparing", 5, "检查输入文件")
            ffmpeg_path = find_ffmpeg()
            prepared = prepare_audio_source(
                source_path=source_path,
                use_vad=self.settings.vad.enabled,
                ffmpeg_path=ffmpeg_path,
            )
            cleanup.extend(prepared.cleanup_paths)
            if prepared.audio_path != source_path:
                report("Extracting", 40, "音频提取完成")
            else:
                report("Preparing", 35, "输入文件为音频")

            if cancel_event.is_set():
                raise TaskCancelled("转写前已取消")

            provider = self._build_transcription_provider()
            chunks = [AudioChunk(path=prepared.audio_path, start_offset=0.0, end_offset=0.0)]
            if self.settings.vad.enabled:
                report("Preparing", 45, "Silero VAD 预切分中")
                temp_dir = Path(tempfile.mkdtemp(prefix="subtitle_vad_"))
                cleanup.append(temp_dir)
                chunks = split_audio_with_vad(prepared.audio_path, self.settings.vad, temp_dir)
                report("Transcribing", 55, f"VAD 已切分为 {len(chunks)} 段")

            result = self._run_transcription_chunks(source_path, provider, chunks, report, cancel_event)
            if self.settings.segmentation.enabled:
                report("Transcribing", 82, "正在执行智能分段")
                result = self._apply_intelligent_segmentation(result, cancel_event)

            report("Writing", 85, "正在生成转写文件")
            if cancel_event.is_set():
                raise TaskCancelled("写入前已取消")

            target_dir = source_path.parent if self.settings.output.mode == "source" else self.settings.output.output_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            lang_code = (
                normalize_language_code(self.settings.transcription.language)
                if self.settings.transcription.language_mode == "manual"
                else normalize_language_code(result.language or detect_language_code(result.raw_payload))
            ) or "und"

            outputs = write_transcription_outputs(
                target_dir=target_dir,
                source_stem=source_path.stem,
                lang_code=lang_code,
                save_srt=self.settings.output.save_srt,
                save_lrc=self.settings.output.save_lrc,
                save_txt=self.settings.output.save_txt,
                save_json=self.settings.output.save_json,
                is_audio_input=prepared.is_audio_input,
                payload=result.raw_payload,
                segments=result.segments,
                text=result.text,
            )

            if self.settings.translation.mode != "none":
                report("Translating", 90, "正在翻译字幕")
                translated_segments, translated_text = self._translate_transcription_result(result, cancel_event)
                report("Writing", 96, "正在写入翻译文件")
                outputs.update(
                    write_translation_outputs(
                        target_dir=target_dir,
                        source_stem=trim_language_suffix_from_stem(source_path.stem, lang_code),
                        source_lang_code=lang_code,
                        settings=self.settings,
                        original_segments=[dict(seg) for seg in result.segments],
                        translated_segments=translated_segments,
                        original_text=result.text,
                        translated_text=translated_text,
                        source_path=source_path,
                        write_lrc=self.settings.output.save_lrc and prepared.is_audio_input,
                    )
                )

            report("Completed", 100, "完成")
            return outputs
        finally:
            cleanup_paths(cleanup)

    def run_subtitle_translation_task(
        self,
        source_path: Path,
        report: Callable[[str, int, str], None],
        cancel_event: Event,
    ) -> Dict[str, str]:
        if cancel_event.is_set():
            raise TaskCancelled("启动前已取消")
        if self.settings.translation.mode == "none":
            raise RuntimeError("字幕导入任务需要启用翻译模式")

        report("Preparing", 20, "读取字幕文件")
        original_segments, original_text = extract_subtitle_source(source_path)
        if original_segments:
            lines = [str(seg.get("text", "")).strip() for seg in original_segments if str(seg.get("text", "")).strip()]
        else:
            lines = [line.strip() for line in original_text.splitlines() if line.strip()]
            if not lines and original_text.strip():
                lines = [original_text.strip()]
        if not lines:
            raise RuntimeError("字幕文件没有可翻译内容")

        provider = self._build_translation_provider()
        if provider is None:
            raise RuntimeError("字幕导入任务需要启用翻译模式")

        report("Translating", 75, "正在翻译字幕")
        translated_lines = provider.translate_lines(
            lines=lines,
            request=TranslationRequest(
                model=self.settings.translation.model,
                target_language=self.settings.translation.target_language,
            ),
            cancel_event=cancel_event,
            parallel_workers=self.settings.translation.subtitle_translation_thread_count,
        )
        translated_segments: list[dict[str, Any]] = []
        translated_text = "\n".join(translated_lines).strip()
        if original_segments:
            translated_segments = [dict(seg) for seg in original_segments]
            for seg, translated_line in zip(translated_segments, translated_lines):
                seg["text"] = translated_line

        if cancel_event.is_set():
            raise TaskCancelled("写入前已取消")

        target_dir = source_path.parent if self.settings.output.mode == "source" else self.settings.output.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        source_lang_code = (
            normalize_language_code(self.settings.transcription.language)
            if self.settings.transcription.language_mode == "manual"
            else infer_language_code_from_filename(source_path)
        ) or "und"

        report("Writing", 92, "正在写入翻译文件")
        outputs = write_translation_outputs(
            target_dir=target_dir,
            source_stem=trim_language_suffix_from_stem(source_path.stem, source_lang_code),
            source_lang_code=source_lang_code,
            settings=self.settings,
            original_segments=original_segments,
            translated_segments=translated_segments,
            original_text=original_text,
            translated_text=translated_text,
            source_path=source_path,
        )
        report("Completed", 100, "完成")
        return outputs

    def _run_transcription_chunks(
        self,
        source_path: Path,
        provider: WhisperOpenAICompatibleProvider,
        chunks: List[AudioChunk],
        report: Callable[[str, int, str], None],
        cancel_event: Event,
    ) -> TranscriptionResult:
        merged_segments: list[dict[str, Any]] = []
        merged_text_parts: list[str] = []
        first_payload: Optional[dict[str, Any]] = None
        detected_language = ""

        for idx, chunk in enumerate(chunks, start=1):
            if cancel_event.is_set():
                raise TaskCancelled("转写前已取消")
            progress = 60 if len(chunks) == 1 else 55 + int(25 * (idx - 1) / max(len(chunks), 1))
            message = "正在调用转写接口" if len(chunks) == 1 else f"正在转写第 {idx}/{len(chunks)} 段"
            report("Transcribing", progress, message)
            result = provider.transcribe(
                TranscriptionRequest(
                    source_path=source_path,
                    audio_path=chunk.path,
                    language_mode=self.settings.transcription.language_mode,
                    language=self.settings.transcription.language,
                    timestamp_granularity=self.settings.transcription.timestamp_granularity,
                    context_bias=self.settings.transcription.context_bias,
                ),
                progress_cb=None,
                cancel_event=cancel_event,
            )
            if first_payload is None:
                first_payload = dict(result.raw_payload)
            if result.language and not detected_language:
                detected_language = result.language

            cleaned_text = sanitize_transcribed_text(result.text)
            if cleaned_text:
                merged_text_parts.append(cleaned_text)

            if result.segments:
                for segment in result.segments:
                    segment_text = str(segment.get("text", ""))
                    if not segment_text.strip():
                        continue
                    adjusted = dict(segment)
                    adjusted["start"] = float(segment["start"]) + chunk.start_offset
                    adjusted["end"] = float(segment["end"]) + chunk.start_offset
                    adjusted["text"] = segment_text
                    segment_words = segment.get("words")
                    if isinstance(segment_words, list):
                        adjusted_words: list[dict[str, Any]] = []
                        for word in segment_words:
                            if not isinstance(word, dict):
                                continue
                            text = str(word.get("text", ""))
                            if not text or not text.strip():
                                continue
                            try:
                                start = float(word["start"]) + chunk.start_offset
                                end = float(word["end"]) + chunk.start_offset
                            except Exception:
                                continue
                            adjusted_words.append(
                                {
                                    "start": start,
                                    "end": max(start, end),
                                    "text": text,
                                }
                            )
                        if adjusted_words:
                            adjusted["words"] = adjusted_words
                    merged_segments.append(adjusted)
            elif cleaned_text:
                merged_segments.append(
                    {
                        "start": chunk.start_offset,
                        "end": max(chunk.end_offset, chunk.start_offset),
                        "text": cleaned_text,
                    }
                )

        payload = first_payload or {"text": ""}
        payload["segments"] = merged_segments
        payload["text"] = "\n".join(merged_text_parts).strip()
        if detected_language:
            payload["language"] = detected_language
        if (
            isinstance(provider, WhisperOpenAICompatibleProvider)
            and not merged_segments
            and not payload["text"]
        ):
            detail = summarize_empty_transcription_response(
                payload if first_payload is not None else {},
                str(payload.get("text", "")),
            )
            raise RuntimeError(
                "Whisper 兼容接口返回了空转写结果。"
                "这通常表示上游模型或服务虽然返回了 HTTP 200，但实际转写失败。"
                f"{detail}"
            )
        return TranscriptionResult(
            text=payload["text"],
            segments=merged_segments,
            language=detected_language or detect_language_code(payload),
            raw_payload=payload,
        )

    def _apply_intelligent_segmentation(
        self,
        result: TranscriptionResult,
        cancel_event: Event,
    ) -> TranscriptionResult:
        if not self.settings.segmentation.enabled:
            return result
        if self.settings.transcription.timestamp_granularity != "word":
            raise RuntimeError("智能分段仅支持词级时间戳输出，请先将时间戳粒度切换为 word")

        provider = self._build_segmentation_provider()
        if provider is None:
            return result

        words = collect_word_tokens(result.segments)
        if not words:
            raise RuntimeError(
                "智能分段需要词级时间戳，但当前转写结果未包含 words 数据。"
                "请确认所选服务端支持 `word` 时间戳输出。"
            )

        if cancel_event.is_set():
            raise TaskCancelled("智能分段前已取消")

        ranges = provider.segment_words(
            words=words,
            request=SegmentationRequest(
                model=self.settings.segmentation.model,
                source_language=result.language or detect_language_code(result.raw_payload),
            ),
            cancel_event=cancel_event,
        )
        rebuilt_segments = self._rebuild_segments_from_word_ranges(words, ranges)
        if not rebuilt_segments:
            raise RuntimeError("智能分段未生成可用的字幕段落")

        updated_payload = dict(result.raw_payload)
        updated_payload["segments"] = rebuilt_segments
        updated_payload["text"] = "\n".join(
            str(segment.get("text", "")).strip() for segment in rebuilt_segments if str(segment.get("text", "")).strip()
        ).strip()
        updated_payload["intelligent_segmentation"] = {
            "enabled": True,
            "model": self.settings.segmentation.model,
            "word_count": len(words),
            "segment_count": len(rebuilt_segments),
        }

        return TranscriptionResult(
            text=updated_payload["text"],
            segments=rebuilt_segments,
            language=result.language or detect_language_code(updated_payload),
            raw_payload=updated_payload,
        )

    def _rebuild_segments_from_word_ranges(
        self,
        words: list[dict[str, Any]],
        ranges: list[tuple[int, int]],
    ) -> list[dict[str, Any]]:
        rebuilt_segments: list[dict[str, Any]] = []
        for start_index, end_index in ranges:
            bucket = words[start_index : end_index + 1]
            if not bucket:
                continue
            segment_text = sanitize_transcribed_text(join_word_tokens(bucket))
            if not segment_text:
                segment_text = join_word_tokens(bucket)
            if not segment_text:
                continue
            rebuilt_segments.append(
                {
                    "start": float(bucket[0]["start"]),
                    "end": float(bucket[-1]["end"]),
                    "text": segment_text,
                    "words": [dict(word) for word in bucket],
                }
            )
        return rebuilt_segments

    def _translate_transcription_result(
        self,
        result: TranscriptionResult,
        cancel_event: Event,
    ) -> tuple[list[dict[str, Any]], str]:
        provider = self._build_translation_provider()
        if provider is None:
            return [], ""
        lines = [str(seg.get("text", "")).strip() for seg in result.segments if str(seg.get("text", "")).strip()]
        translated_segments: list[dict[str, Any]] = []
        translated_text = ""
        if lines:
            translated_lines = provider.translate_lines(
                lines=lines,
                request=TranslationRequest(
                    model=self.settings.translation.model,
                    target_language=self.settings.translation.target_language,
                ),
                cancel_event=cancel_event,
                parallel_workers=self.settings.translation.subtitle_translation_thread_count,
            )
            translated_segments = [dict(seg) for seg in result.segments]
            for seg, translated_line in zip(translated_segments, translated_lines):
                seg["text"] = translated_line
            translated_text = "\n".join(translated_lines).strip()
        elif result.text.strip():
            translated_lines = provider.translate_lines(
                lines=[result.text],
                request=TranslationRequest(
                    model=self.settings.translation.model,
                    target_language=self.settings.translation.target_language,
                ),
                cancel_event=cancel_event,
                parallel_workers=self.settings.translation.subtitle_translation_thread_count,
            )
            translated_text = translated_lines[0] if translated_lines else ""
        return translated_segments, translated_text

    def _build_transcription_provider(
        self,
    ) -> WhisperOpenAICompatibleProvider:
        return WhisperOpenAICompatibleProvider(self.settings.transcription.whisper)

    def _build_translation_provider(self):
        return build_translation_provider(
            mode=self.settings.translation.mode,
            openai_base_url=self.settings.translation.openai_base_url,
            openai_api_key=self.settings.translation.openai_api_key,
            thinking_enabled=self.settings.translation.thinking_enabled,
            reasoning_effort=self.settings.translation.reasoning_effort,
            temperature=self.settings.translation.temperature,
            chunk_size=self.settings.translation.chunk_size,
        )

    def _build_segmentation_provider(self):
        return build_segmentation_provider(self.settings.segmentation)
