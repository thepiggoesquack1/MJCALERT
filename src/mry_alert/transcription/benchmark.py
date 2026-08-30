from __future__ import annotations

import asyncio
import gc
import logging
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from mry_alert.adsb.mock import MockNearbyAircraftProvider
from mry_alert.audio.file_source import WaveFileAudioSource
from mry_alert.audio.segmenter import VadSegmenter
from mry_alert.config import AppConfig, SpeechConfig
from mry_alert.detection.engine import DetectionEngine, DetectionEvent
from mry_alert.models import TranscriptEvent
from mry_alert.transcription.base import Transcriber
from mry_alert.transcription.evaluation import (
    classify_rtf,
    evaluation_metrics,
    extract_aviation_entities,
)
from mry_alert.transcription.faster_whisper import FasterWhisperTranscriber

TranscriberFactory = Callable[[SpeechConfig, Any], Transcriber]
logger = logging.getLogger(__name__)


def cleanup_transcriber(transcriber: object) -> None:
    started = time.perf_counter()
    model = getattr(transcriber, "model", None)
    for target in (model, transcriber):
        for method_name in ("close", "unload"):
            method = getattr(target, method_name, None)
            if callable(method):
                try:
                    method()
                except (RuntimeError, OSError, ValueError) as exc:
                    logger.warning("Model cleanup method %s failed: %s", method_name, exc)
                break
    del model
    gc.collect()
    logger.info(
        "Benchmark timing: model cleanup completed in %.3f seconds", time.perf_counter() - started
    )


async def load_segmented_wave(path: Path, config: AppConfig) -> tuple[float, list[bytes]]:
    source = WaveFileAudioSource(path, config.audio.frame_duration_ms)
    segmenter = VadSegmenter(config.audio)
    segments: list[bytes] = []
    duration = 0.0
    async for frame in source.frames():
        duration += len(frame) / 2 / config.audio.sample_rate
        if transmission := segmenter.add_frame(frame):
            segments.append(transmission)
    if transmission := segmenter.flush():
        segments.append(transmission)
    return duration, segments


async def benchmark_model(
    model_name: str,
    segments: list[bytes],
    audio_duration: float,
    config: AppConfig,
    expected_transcript: str | None = None,
    *,
    transcriber_factory: TranscriberFactory = FasterWhisperTranscriber,
    send_notifications: bool = False,
) -> dict[str, Any]:
    speech = config.speech_for_model(model_name)
    tracemalloc.start()
    logger.info("Benchmark timing: constructing model %s", model_name)
    initialized = time.perf_counter()
    transcriber = transcriber_factory(speech, config.audio.preprocessing)
    initialization_seconds = time.perf_counter() - initialized
    emitted: list[DetectionEvent] = []

    async def publish(event: DetectionEvent) -> None:
        emitted.append(event)

    evaluation_config = config.model_copy(deep=True)
    evaluation_config.detection.destination_confirmation_delay_seconds = 0
    engine = DetectionEngine(evaluation_config, MockNearbyAircraftProvider([]), publisher=publish)
    started = time.perf_counter()
    segment_results: list[dict[str, Any]] = []
    transcript_parts: list[str] = []
    base_time = datetime.now(UTC)
    for index, pcm in enumerate(segments):
        duration = len(pcm) / 2 / config.audio.sample_rate
        segment_started = time.perf_counter()
        logger.info("Benchmark timing: model=%s segment=%d transcribe entry", model_name, index)
        result = await asyncio.to_thread(transcriber.transcribe, pcm, config.audio.sample_rate)
        processing_seconds = time.perf_counter() - segment_started
        transcript_parts.append(result.text)
        event = TranscriptEvent(
            event_id=str(uuid4()),
            timestamp=base_time + timedelta(seconds=index),
            text=result.text,
            source=config.liveatc.source_label,
            duration_seconds=duration,
            transcription_confidence=result.language_probability,
            average_log_probability=result.average_log_probability,
            no_speech_probability=result.no_speech_probability,
            transcription_segment_count=result.segment_count,
            transcript_duration_seconds=result.duration_seconds,
            whisper_quality=result.quality,
        )
        detector_started = time.perf_counter()
        detector_result = await engine.process(event)
        logger.info(
            "Benchmark timing: detector processing completed in %.3f seconds",
            time.perf_counter() - detector_started,
        )
        aviation_entities = extract_aviation_entities(result.text, config.destinations)
        segment_results.append(
            {
                "index": index,
                "audio_duration_seconds": duration,
                "processing_seconds": processing_seconds,
                "real_time_factor": processing_seconds / duration if duration else None,
                "raw_transcript": result.text,
                "normalized_transcript": event.normalized_text,
                "decoder_confidence": result.quality,
                "average_log_probability": result.average_log_probability,
                "no_speech_probability": result.no_speech_probability,
                "hallucination_filter": {
                    "trimmed": event.artifact_trimming_reason is not None,
                    "reason": event.artifact_trimming_reason,
                },
                "speaker_role": event.speaker_role.value,
                "speaker_role_confidence": event.speaker_role_confidence,
                "speaker_role_reasons": event.speaker_role_reasons,
                "callsign_candidates": [event.detected_callsign] if event.detected_callsign else [],
                "aviation_entities": asdict(aviation_entities),
                "detected_destination": event.destination_candidate,
                "detected_intent": event.intent_category.value,
                "final_detector_decision": event.detection_decision.value,
                "detector_reasons": event.detection_reasons,
                "adsb_identification": event.identified_registration,
                "identification_source": event.identification_source.value,
                "adsb_decision": event.adsb_decision.value if event.adsb_decision else None,
                "adsb_winning_score": event.adsb_winning_score,
                "adsb_winning_margin": event.adsb_winning_margin,
                "correction_handling": detector_result.event_type.value
                if detector_result
                else None,
                "notification_would_be_sent": detector_result is not None,
                "notification_sent": False,
            }
        )
    await engine.flush_pending()
    await engine.close()
    total_processing = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    transcript = " ".join(part for part in transcript_parts if part).strip()
    avg_logs = [
        item["average_log_probability"]
        for item in segment_results
        if item["average_log_probability"] is not None
    ]
    no_speech = [
        item["no_speech_probability"]
        for item in segment_results
        if item["no_speech_probability"] is not None
    ]
    rtf = total_processing / audio_duration if audio_duration else 0.0
    loaded_model = getattr(transcriber, "loaded_model", model_name)
    report: dict[str, Any] = {
        "status": "success",
        "model": model_name,
        "loaded_model": loaded_model,
        "initialization_seconds": initialization_seconds,
        "total_processing_seconds": total_processing,
        "audio_duration_seconds": audio_duration,
        "real_time_factor": rtf,
        "performance_classification": classify_rtf(
            rtf,
            config.speech_performance.excellent_rtf_max,
            config.speech_performance.acceptable_rtf_max,
            config.speech_performance.risky_rtf_max,
        ),
        "transcript": transcript,
        "decoder_confidence": _aggregate_quality(segment_results),
        "average_log_probability": sum(avg_logs) / len(avg_logs) if avg_logs else None,
        "no_speech_probability": sum(no_speech) / len(no_speech) if no_speech else None,
        "static_prompt_enabled": speech.use_static_aviation_prompt,
        "adsb_dynamic_prompt_enabled": speech.use_adsb_dynamic_prompt
        and speech.adsb_prompt.enabled,
        "adsb_dynamic_prompt_used": False,
        "prompted_nearby_registrations": [],
        "peak_python_memory_mb": peak / (1024 * 1024),
        "peak_gpu_memory_mb": None,
        "segments": segment_results,
        "notification_delivery_enabled": send_notifications,
        "notifications_sent": 0,
    }
    if expected_transcript is not None:
        report["expected_transcript_metrics"] = evaluation_metrics(
            expected_transcript, transcript, config.destinations
        )
    cleanup_transcriber(transcriber)
    del transcriber
    return report


def _aggregate_quality(segments: list[dict[str, Any]]) -> str | None:
    qualities = [str(item["decoder_confidence"]) for item in segments if item["decoder_confidence"]]
    if not qualities:
        return None
    rank = {"low": 0, "medium": 1, "high": 2}
    return min(qualities, key=lambda quality: rank.get(quality, 0))
