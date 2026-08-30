from __future__ import annotations

import asyncio
import logging
import time
import wave
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import uuid4

from fastapi import WebSocket

from mry_alert.adsb.base import NearbyAircraftProvider
from mry_alert.adsb.factory import create_nearby_aircraft_provider
from mry_alert.audio.segmenter import VadSegmenter
from mry_alert.audio_classifier import AudioIntentClassifier, create_audio_classifier
from mry_alert.audio_classifier.dataset import TrainingDataCollector
from mry_alert.audio_classifier.models import (
    AudioIntentResult,
    ClassificationContext,
    DestinationLabel,
    IntentLabel,
)
from mry_alert.config import AppConfig, AudioConfig
from mry_alert.detection.engine import DetectionEngine, DetectionEvent
from mry_alert.models import (
    ConfirmationStatus,
    DestinationIntentCategory,
    DetectionDecision,
    PendingDestinationEvent,
    TranscriptEvent,
)
from mry_alert.operational_logging import log_pending_cancelled, log_transmission_result
from mry_alert.server.runtime_status import RuntimeStatus, utc_now
from mry_alert.transcription.adsb_prompt import AdsbPromptCache
from mry_alert.transcription.base import Transcriber
from mry_alert.transcription.faster_whisper import FasterWhisperTranscriber

logger = logging.getLogger(__name__)
AlertPublisher = Callable[[DetectionEvent], Awaitable[None]]
OutcomeObserver = Callable[[TranscriptEvent], None]


class AudioWebSocketService(Protocol):
    status: str

    async def handle(self, websocket: WebSocket) -> None: ...


class TransmissionSegmenter(Protocol):
    def add_frame(self, frame: bytes) -> bytes | None: ...

    def flush(self) -> bytes | None: ...


class LiveAudioIngestService:
    """Consume transient 16 kHz PCM and run the existing local detection pipeline."""

    def __init__(
        self,
        config: AppConfig,
        publish: AlertPublisher,
        transcriber: Transcriber | None = None,
        provider: NearbyAircraftProvider | None = None,
        segmenter_factory: Callable[[AudioConfig], TransmissionSegmenter] = VadSegmenter,
        runtime_status: RuntimeStatus | None = None,
        classifier: AudioIntentClassifier | None = None,
        outcome_observer: OutcomeObserver | None = None,
        lifecycle_publish: AlertPublisher | None = None,
    ) -> None:
        self.config = config
        self.publish = publish
        self._transcriber = transcriber
        self._model_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()
        raw_provider = provider or create_nearby_aircraft_provider(
            config.airport, config.adsb
        )
        self._engine = DetectionEngine(
            config,
            raw_provider,
            publisher=publish,
            lifecycle_publisher=lifecycle_publish,
        )
        self._provider = self._engine.provider
        self._adsb_prompt = AdsbPromptCache()
        self._segmenter_factory = segmenter_factory
        self._recent_rtfs: deque[float] = deque(maxlen=config.speech_performance.rolling_window)
        self.processing_backlog = 0
        self.runtime_status = runtime_status
        self._outcome_observer = outcome_observer
        self._classifier = classifier
        self._classifier_error: str | None = None
        if self._classifier is None:
            try:
                self._classifier = create_audio_classifier(config)
            except RuntimeError as exc:
                self._classifier_error = str(exc)
                logger.error("Audio classifier load failed: %s", exc)
        self._collector = TrainingDataCollector(config.training_data)
        if self.runtime_status:
            self.runtime_status.classifier_enabled = config.audio_classifier.enabled
            self.runtime_status.classifier_loaded = (
                config.audio_classifier.enabled and self._classifier is not None
            )
            self.runtime_status.classifier_model_version = (
                self._classifier.model_version if self._classifier else None
            )
            self.runtime_status.classifier_error = self._classifier_error
            self.runtime_status.dataset_collection_enabled = config.training_data.enabled
            self.runtime_status.pending_review_clips = self._collector.pending_count()
        if self.runtime_status and transcriber is not None:
            self.runtime_status.speech_ready = True
        self.status = "idle"
        logger.info(
            "Speech runtime model=%s language=%s beam=%d temperature=%g "
            "previous_text=%s internal_vad=%s preprocessing=%s adsb_prompt=%s",
            config.speech.model,
            config.speech.language,
            config.speech.beam_size,
            config.speech.temperature,
            config.speech.condition_on_previous_text,
            config.speech.vad_filter,
            config.audio.preprocessing.enabled,
            config.speech.adsb_prompt.enabled,
        )

    async def _refresh_adsb_prompt(self) -> None:
        while True:
            try:
                nearby = await self._provider.nearby()
                self._engine.observe_adsb(nearby)
                if self.config.speech.adsb_prompt.enabled:
                    self._adsb_prompt.update(
                        nearby, self.config.speech.adsb_prompt.max_aircraft
                    )
            except Exception as exc:
                logger.warning("ADS-B transcription prompt refresh failed: %s", exc)
            await asyncio.sleep(self.config.speech.adsb_prompt.refresh_seconds)

    async def _get_transcriber(self) -> Transcriber:
        if self._transcriber is not None:
            return self._transcriber
        async with self._model_lock:
            if self._transcriber is None:
                self.status = "loading_model"
                self._transcriber = await asyncio.to_thread(
                    FasterWhisperTranscriber,
                    self.config.speech_for_model(self.config.speech.model),
                    self.config.audio.preprocessing,
                )
                if self.runtime_status:
                    self.runtime_status.speech_ready = True
        return self._transcriber

    async def _process_transmission(
        self, websocket: WebSocket, transcriber: Transcriber, pcm: bytes, queue_depth: int = 0
    ) -> None:
        classifier_status: Literal["available", "disabled", "unavailable", "failed"] = (
            "disabled" if not self.config.audio_classifier.enabled else "unavailable"
        )
        classification = (
            AudioIntentResult(
                model_version="unavailable",
                reasons=[self._classifier_error or "audio classifier unavailable"],
            )
            if self.config.audio_classifier.enabled
            else None
        )
        if self.config.audio_classifier.enabled and self._classifier is not None:
            classifier_status = "available"
            context = ClassificationContext(
                timestamp=datetime.now(UTC),
                source=self.config.liveatc.source_label,
                nearby_registrations=self._adsb_prompt.registrations,
            )
            try:
                classification = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._classifier.classify,
                        pcm,
                        self.config.audio.sample_rate,
                        context,
                    ),
                    timeout=self.config.audio_classifier.inference_timeout_seconds,
                )
                logger.info(
                    "AUDIO CLASSIFIER\nDestination: %s\nDestination confidence: %.3f\n"
                    "Intent: %s\nIntent confidence: %.3f\nCorrection: %s\n"
                    "Noise probability: %.3f\nModel: %s",
                    classification.destination.value,
                    classification.destination_confidence,
                    classification.intent.value,
                    classification.intent_confidence,
                    classification.correction,
                    classification.noise_confidence,
                    classification.model_version,
                )
                if self.runtime_status:
                    self.runtime_status.last_classification_at = utc_now()
                    self.runtime_status.last_classification = classification.model_dump(
                        exclude={"raw_scores"}
                        if not self.config.audio_classifier.log_raw_scores
                        else set(),
                        mode="json",
                    )
            except TimeoutError:
                logger.error("Audio classifier inference timed out")
                classifier_status = "failed"
                classification = AudioIntentResult(
                    model_version=self._classifier.model_version,
                    reasons=["audio classifier inference timed out"],
                )
            except Exception as exc:
                logger.exception("Audio classifier inference failed: %s", exc)
                classifier_status = "failed"
                classification = AudioIntentResult(
                    model_version=self._classifier.model_version,
                    reasons=["audio classifier inference failed"],
                )
        set_prompt = getattr(transcriber, "set_dynamic_prompt", None)
        if callable(set_prompt):
            set_prompt(self._adsb_prompt.prompt)
        started = time.perf_counter()
        transcription = await asyncio.to_thread(
            transcriber.transcribe, pcm, self.config.audio.sample_rate
        )
        processing_seconds = time.perf_counter() - started
        audio_seconds = len(pcm) / 2 / self.config.audio.sample_rate
        rtf = processing_seconds / audio_seconds if audio_seconds else 0.0
        self._recent_rtfs.append(rtf)
        average_rtf = sum(self._recent_rtfs) / len(self._recent_rtfs)
        self.processing_backlog = queue_depth
        if average_rtf > self.config.speech_performance.live_warning_rtf:
            logger.warning(
                "LIVE TRANSCRIPTION WARNING\nAverage RTF: %.2f\nQueued segments: %d\n"
                "Model may not keep up with current traffic",
                average_rtf,
                queue_depth,
            )
        text = transcription.text
        confidence = transcription.language_probability
        if not text.strip() and classification is None:
            return
        if self.runtime_status:
            self.runtime_status.last_transcription_at = utc_now()
            self.runtime_status.last_transcript = text
        event = TranscriptEvent(
            event_id=str(uuid4()),
            timestamp=datetime.now(UTC),
            text=text or "[no reliable Whisper transcript]",
            source=self.config.liveatc.source_label,
            duration_seconds=len(pcm) / 2 / self.config.audio.sample_rate,
            transcription_confidence=confidence,
            average_log_probability=transcription.average_log_probability,
            no_speech_probability=transcription.no_speech_probability,
            transcription_segment_count=transcription.segment_count,
            transcript_duration_seconds=transcription.duration_seconds,
            whisper_quality=transcription.quality,
            audio_intent=classification,
        )
        destination_event = await self._engine.process(event)
        if destination_event is None and self._outcome_observer is not None:
            self._outcome_observer(event)
        if self.runtime_status:
            self.runtime_status.last_detected_callsign = event.detected_callsign
            self.runtime_status.last_aircraft_type = (
                event.aircraft_type_name or event.aircraft_type_code
            )
            self.runtime_status.last_destination = event.destination_candidate
            self.runtime_status.last_detector_decision = event.detection_decision.value
            self.runtime_status.recent_intent_result = (
                f"likely {event.destination_candidate}"
                if event.destination_candidate
                else event.intent_category.value
            )
            self.runtime_status.recent_intent_decision = {
                DetectionDecision.PENDING: "pending ADS-B confirmation",
                DetectionDecision.CONFIRMED: "notification confirmed",
                DetectionDecision.UNRESOLVED: "unresolved",
                DetectionDecision.AMBIGUOUS: "ambiguous",
                DetectionDecision.IGNORED: "ignored",
                DetectionDecision.CANCELLED: "cancelled",
                DetectionDecision.CORRECTED: "corrected",
            }.get(event.detection_decision, event.detection_decision.value.lower())
            if event.adsb_decision is not None:
                self.runtime_status.last_adsb_correlation_at = utc_now()
                self.runtime_status.last_adsb_winner = event.identified_registration
                self.runtime_status.last_adsb_score = event.adsb_winning_score
                self.runtime_status.last_adsb_margin = event.adsb_winning_margin
        if event.traffic_filter_decision != "ignored_scheduled_airline":
            classifier_candidate = bool(
                classification
                and (
                    classification.destination != DestinationLabel.NONE
                    or classification.intent not in {IntentLabel.NONE, IntentLabel.NOISE}
                    or classification.noise_confidence
                    >= self.config.audio_classifier.noise_rejection_threshold
                )
            )
            detector_candidate = bool(
                event.destination_candidate
                or event.intent_category != DestinationIntentCategory.NONE
                or event.detection_decision
                in {
                    DetectionDecision.PENDING,
                    DetectionDecision.CONFIRMED,
                    DetectionDecision.UNRESOLVED,
                    DetectionDecision.AMBIGUOUS,
                    DetectionDecision.CANCELLED,
                    DetectionDecision.CORRECTED,
                }
            )
            candidate = bool(
                self.config.training_data.save_all_candidate_events
                or classifier_candidate
                or detector_candidate
                or (classifier_status != "available" and bool(text.strip()))
            )
            if candidate:
                uncertain = (
                    classification is None
                    or classifier_status != "available"
                    or destination_event is None
                    or classification.destination_confidence
                    < self.config.audio_classifier.strong_confidence_threshold
                    or classification.intent_confidence
                    < self.config.audio_classifier.strong_confidence_threshold
                )
                try:
                    saved = self._collector.save(
                        pcm,
                        self.config.audio.sample_rate,
                        transcript=text,
                        normalized_transcript=event.normalized_text,
                        classification=classification,
                        classifier_status=classifier_status,
                        detection_event=event,
                        adsb_candidates=[
                            {"reason": reason} for reason in event.adsb_candidate_reasons
                        ],
                        selected_aircraft=event.identified_registration,
                        adsb_supported=event.identified_registration is not None,
                        uncertain=uncertain,
                    )
                    if saved and self.runtime_status:
                        self.runtime_status.pending_review_clips = self._collector.pending_count()
                        self.runtime_status.training_clip_last_error = None
                except (OSError, ValueError, wave.Error) as exc:
                    logger.error("Training clip collection failed: %s", exc)
                    if self.runtime_status:
                        self.runtime_status.training_clip_failures += 1
                        self.runtime_status.training_clip_last_error = str(exc)
        log_transmission_result(event, self.config.logging)
        if isinstance(destination_event, PendingDestinationEvent) and (
            destination_event.confirmation_status
            in {ConfirmationStatus.CORRECTED, ConfirmationStatus.CANCELLED}
        ):
            log_pending_cancelled(destination_event, self.config.logging)
        payload = {
            "type": "transcript",
            "text": text,
            "confidence": confidence,
            "decoder_confidence": transcription.quality,
            "decoder_confidence_explanation": (
                "Decoder confidence reflects token-sequence confidence and does not "
                "guarantee transcription accuracy."
            ),
            "destination_event_created": destination_event is not None,
            "alert_created": False,
            "speaker_role": event.speaker_role,
            "speaker_role_confidence": event.speaker_role_confidence,
            "speaker_role_reasons": event.speaker_role_reasons,
            "timestamp": event.timestamp.isoformat(),
            "real_time_factor": rtf,
            "recent_average_rtf": average_rtf,
            "processing_backlog": queue_depth,
        }
        try:
            await websocket.send_json(payload)
        except RuntimeError:
            logger.info("Audio client disconnected before transcript status delivery")

    async def handle(self, websocket: WebSocket) -> None:
        if self._session_lock.locked():
            await websocket.accept()
            await websocket.send_json(
                {"type": "error", "message": "Another LiveATC capture session is active."}
            )
            await websocket.close(code=1013)
            return
        async with self._session_lock:
            prompt_task: asyncio.Task[None] | None = None
            await websocket.accept()
            self.status = "initializing"
            await websocket.send_json({"type": "status", "status": self.status})
            try:
                if self._transcriber is None:
                    self.status = "loading_model"
                    await websocket.send_json({"type": "status", "status": self.status})
                transcriber = await self._get_transcriber()
                segmenter = self._segmenter_factory(self.config.audio)
            except (RuntimeError, OSError, ValueError) as exc:
                self.status = "error"
                logger.error("Live audio initialization failed: %s", exc)
                await websocket.send_json({"type": "error", "message": str(exc)})
                await websocket.close(code=1011)
                return
            if self.config.adsb_tracking.enabled or (
                self.config.speech.adsb_prompt.enabled
                and self.config.speech.use_adsb_dynamic_prompt
            ):
                prompt_task = asyncio.create_task(
                    self._refresh_adsb_prompt(), name="refresh-adsb-context"
                )

            frame_bytes = (
                self.config.audio.sample_rate * self.config.audio.frame_duration_ms // 1000 * 2
            )
            pending = bytearray()
            self.status = "monitoring"
            await websocket.send_json({"type": "status", "status": self.status})
            transmission_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

            async def process_queue() -> None:
                while True:
                    queued = await transmission_queue.get()
                    try:
                        if queued is None:
                            return
                        self.processing_backlog = transmission_queue.qsize()
                        await self._process_transmission(
                            websocket,
                            transcriber,
                            queued,
                            transmission_queue.qsize(),
                        )
                    finally:
                        transmission_queue.task_done()

            worker = asyncio.create_task(process_queue(), name="live-transcription-worker")
            try:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        break
                    chunk = message.get("bytes")
                    if not isinstance(chunk, bytes):
                        continue
                    if self.runtime_status:
                        self.runtime_status.last_audio_received_at = utc_now()
                    pending.extend(chunk)
                    while len(pending) >= frame_bytes:
                        frame = bytes(pending[:frame_bytes])
                        del pending[:frame_bytes]
                        transmission = segmenter.add_frame(frame)
                        if transmission:
                            transmission_queue.put_nowait(transmission)
                            self.processing_backlog = transmission_queue.qsize()
            finally:
                if transmission := segmenter.flush():
                    transmission_queue.put_nowait(transmission)
                transmission_queue.put_nowait(None)
                try:
                    await worker
                except RuntimeError:
                    logger.info("Audio client disconnected before final transcript status")
                self.status = "idle"
                self.processing_backlog = 0
                if prompt_task:
                    prompt_task.cancel()
                    await asyncio.gather(prompt_task, return_exceptions=True)
