from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import WebSocket

from mry_alert.adsb.mock import MockNearbyAircraftProvider
from mry_alert.audio_classifier.base import MockAudioIntentClassifier
from mry_alert.audio_classifier.dataset import load_metadata
from mry_alert.audio_classifier.models import (
    AudioIntentResult,
    ClassificationContext,
    DestinationLabel,
    IntentLabel,
)
from mry_alert.config import (
    AppConfig,
    AudioClassifierConfig,
    AudioConfig,
    DetectionConfig,
    TrainingDataConfig,
)
from mry_alert.detection.engine import DetectionEvent
from mry_alert.models import AlertEvent, NearbyAircraft, PendingDestinationEvent
from mry_alert.server.audio_ingest import LiveAudioIngestService
from mry_alert.transcription.mock import MockTranscriber


class ImmediateSegmenter:
    def __init__(self, config: AudioConfig) -> None:
        del config

    def add_frame(self, frame: bytes) -> bytes | None:
        return frame

    def flush(self) -> bytes | None:
        return None


class FakeWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self.sent: list[dict[str, Any]] = []
        self._incoming = [
            {"type": "websocket.receive", "bytes": b"\0" * 960},
            {"type": "websocket.disconnect"},
        ]

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, value: dict[str, Any]) -> None:
        self.sent.append(value)

    async def receive(self) -> dict[str, Any]:
        return self._incoming.pop(0)

    async def close(self, code: int = 1000) -> None:
        del code


class HangingClassifier:
    model_version = "hanging-test"

    def classify(
        self, pcm: bytes, sample_rate: int, context: ClassificationContext
    ) -> AudioIntentResult:
        del pcm, sample_rate, context
        time.sleep(0.1)
        return AudioIntentResult()


@pytest.mark.asyncio
async def test_pcm_ingest_runs_local_detection_and_publishes_alert(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="mry_alert.operations")
    published: list[DetectionEvent] = []

    async def publish(alert: DetectionEvent) -> None:
        published.append(alert)

    nearby = NearbyAircraft(
        hex="abc123",
        registration="N123AB",
        flight="N123AB",
        on_ground=True,
        ground_speed=7,
        seconds_since_seen=2,
    )
    service = LiveAudioIngestService(
        AppConfig(detection=DetectionConfig(destination_confirmation_delay_seconds=0.01)),
        publish,
        transcriber=MockTranscriber(
            [
                "Monterey Ground, November one two three alpha bravo, "
                "request taxi to Monterey Jet Center."
            ]
        ),
        provider=MockNearbyAircraftProvider([nearby]),
        segmenter_factory=ImmediateSegmenter,
    )
    websocket = FakeWebSocket()

    await service.handle(cast(WebSocket, websocket))

    assert websocket.accepted
    assert [message["status"] for message in websocket.sent[:2]] == [
        "initializing",
        "monitoring",
    ]
    assert websocket.sent[2]["type"] == "transcript"
    assert websocket.sent[2]["destination_event_created"] is True
    assert websocket.sent[2]["alert_created"] is False
    assert websocket.sent[2]["speaker_role"] == "pilot"
    assert websocket.sent[2]["speaker_role_confidence"] >= 0.8
    assert "first-person taxi request" in websocket.sent[2]["speaker_role_reasons"]
    assert len(published) >= 1
    assert isinstance(published[0], PendingDestinationEvent)
    await asyncio.sleep(0.02)
    confirmed = [item for item in published if isinstance(item, AlertEvent)]
    assert len(confirmed) == 1
    assert confirmed[0].registration == "N123AB"
    transmission_logs = [
        record for record in caplog.records if "ATC TRANSMISSION" in record.getMessage()
    ]
    assert len(transmission_logs) == 1
    assert "Decision:     PENDING" in transmission_logs[0].getMessage()
    assert "Decoder confidence: high" in transmission_logs[0].getMessage()
    assert "request taxi to Monterey Jet Center" in transmission_logs[0].getMessage()
    assert "\\x00" not in caplog.text
    assert service.status == "idle"


@pytest.mark.asyncio
async def test_classifier_timeout_blocks_whisper_only_alert() -> None:
    published: list[DetectionEvent] = []

    async def publish(alert: DetectionEvent) -> None:
        published.append(alert)

    service = LiveAudioIngestService(
        AppConfig(
            audio_classifier=AudioClassifierConfig(enabled=True, inference_timeout_seconds=0.001),
            detection=DetectionConfig(destination_confirmation_delay_seconds=0.01),
        ),
        publish,
        transcriber=MockTranscriber(
            ["November one two three alpha bravo request taxi to Monterey Jet Center"]
        ),
        provider=MockNearbyAircraftProvider([]),
        segmenter_factory=ImmediateSegmenter,
        classifier=HangingClassifier(),
    )

    await service.handle(cast(WebSocket, FakeWebSocket()))

    assert published == []


@pytest.mark.asyncio
async def test_filtered_airline_is_not_saved_as_positive_training_example(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        audio_classifier=AudioClassifierConfig(enabled=True),
        training_data=TrainingDataConfig(
            enabled=True,
            directory=tmp_path / "training",
            save_uncertain_only=False,
            save_all_candidate_events=True,
        ),
    )
    classification = AudioIntentResult(
        destination=DestinationLabel.MONTEREY_JET_CENTER,
        destination_confidence=0.98,
        intent=IntentLabel.TAXI_OR_ROUTE,
        intent_confidence=0.98,
        model_version="mock",
    )
    service = LiveAudioIngestService(
        config,
        publish=lambda _event: asyncio.sleep(0),
        transcriber=MockTranscriber(["United one two three request taxi to Monterey Jet Center"]),
        provider=MockNearbyAircraftProvider(
            [
                NearbyAircraft(
                    hex="ual",
                    flight="UAL123",
                    operator_name="United Airlines",
                    icao_designator="UAL",
                    on_ground=True,
                    seconds_since_seen=1,
                )
            ]
        ),
        segmenter_factory=ImmediateSegmenter,
        classifier=MockAudioIntentClassifier(classification),
    )

    await service.handle(cast(WebSocket, FakeWebSocket()))

    assert not list((tmp_path / "training").rglob("*_metadata.json"))


@pytest.mark.asyncio
async def test_monitoring_collects_review_clip_with_classifier_disabled(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        training_data=TrainingDataConfig(
            enabled=True,
            directory=tmp_path / "training",
            save_audio=True,
        )
    )
    service = LiveAudioIngestService(
        config,
        publish=lambda _event: asyncio.sleep(0),
        transcriber=MockTranscriber(["Radio check, holding short Alpha"]),
        provider=MockNearbyAircraftProvider([]),
        segmenter_factory=ImmediateSegmenter,
    )

    await service.handle(cast(WebSocket, FakeWebSocket()))

    paths = list((tmp_path / "training" / "pending").glob("*_metadata.json"))
    assert len(paths) == 1
    metadata = load_metadata(paths[0])
    assert metadata.classifier_status == "disabled"
    assert metadata.classifier_output.model_version == "unavailable"
    assert metadata.original_transcript == "Radio check, holding short Alpha"
    assert metadata.detection_event_id
    assert metadata.timestamp.tzinfo is not None
    assert metadata.wav_file
    assert (paths[0].parent / metadata.wav_file).is_file()


@pytest.mark.asyncio
async def test_missing_classifier_model_still_collects_review_clip(tmp_path: Path) -> None:
    config = AppConfig(
        audio_classifier=AudioClassifierConfig(
            enabled=True,
            model_path=tmp_path / "model-that-does-not-exist",
        ),
        training_data=TrainingDataConfig(
            enabled=True,
            directory=tmp_path / "training",
        ),
    )
    service = LiveAudioIngestService(
        config,
        publish=lambda _event: asyncio.sleep(0),
        transcriber=MockTranscriber(["November one two three, request taxi"]),
        provider=MockNearbyAircraftProvider([]),
        segmenter_factory=ImmediateSegmenter,
    )

    await service.handle(cast(WebSocket, FakeWebSocket()))

    path = next((tmp_path / "training" / "pending").glob("*_metadata.json"))
    assert load_metadata(path).classifier_status == "unavailable"
