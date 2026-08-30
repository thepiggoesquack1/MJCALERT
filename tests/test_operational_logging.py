from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest
from fastapi import WebSocket

from mry_alert.adsb.mock import MockNearbyAircraftProvider
from mry_alert.config import AppConfig, DetectionConfig, LoggingConfig
from mry_alert.detection.engine import DetectionEngine, DetectionEvent
from mry_alert.models import (
    AlertEvent,
    AlertEventType,
    ConfirmationStatus,
    DetectionDecision,
    MatchStatus,
    NearbyAircraft,
    PendingDestinationEvent,
    SpeakerRole,
    TranscriptEvent,
)
from mry_alert.operational_logging import (
    DeliveryLogTracker,
    DeliveryResult,
    log_notification_delivery,
    log_pending_cancelled,
    log_transmission_result,
)
from mry_alert.server.websocket import WebSocketManager


def alert(**updates: object) -> AlertEvent:
    values: dict[str, object] = {
        "event_id": "event-1",
        "timestamp": datetime.now(UTC),
        "destination": "Monterey Jet Center",
        "registration": "N123AB",
        "spoken_callsign": "Citation 3AB",
        "confidence": 0.93,
        "status": MatchStatus.CONFIRMED,
        "transcript_excerpt": "request taxi to Monterey Jet Center",
    }
    values.update(updates)
    return AlertEvent.model_validate(values)


def transcript() -> TranscriptEvent:
    return TranscriptEvent(
        event_id="transmission-1",
        timestamp=datetime.now(UTC),
        text="Citation three alpha bravo, Monterey Jet Center.",
        normalized_text="citation three alpha bravo monterey jet center",
        speaker_role=SpeakerRole.PILOT,
        speaker_role_confidence=0.82,
        detected_callsign="Citation 3AB",
        destination_candidate="Monterey Jet Center",
        destination_candidate_confidence=0.91,
        detection_decision=DetectionDecision.PENDING,
        detection_reasons=["Waiting 8 seconds for a possible correction"],
    )


def operational_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage() for record in caplog.records if record.name == "mry_alert.operations"
    ]


def test_transmission_result_is_one_readable_log_record(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="mry_alert.operations")
    log_transmission_result(transcript(), LoggingConfig())
    messages = operational_messages(caplog)
    assert len(messages) == 1
    assert 'Heard:        "Citation three alpha bravo' in messages[0]
    assert "Role:         pilot (82%)" in messages[0]
    assert "Decision:     PENDING" in messages[0]


def test_compact_format_contains_heard_and_decision_lines(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="mry_alert.operations")
    log_transmission_result(transcript(), LoggingConfig(format="compact"))
    message = operational_messages(caplog)[0]
    assert '] HEARD "Citation three alpha bravo' in message
    assert "] PENDING Citation 3AB -> Monterey Jet Center" in message


@pytest.mark.asyncio
async def test_confirmation_is_logged_only_after_delay(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="mry_alert.operations")
    settings = LoggingConfig()
    tracker = DeliveryLogTracker()

    async def publish(event: DetectionEvent) -> None:
        if isinstance(event, AlertEvent):
            log_notification_delivery(event, DeliveryResult(1, 1, 0), settings, tracker)

    config = AppConfig(detection=DetectionConfig(destination_confirmation_delay_seconds=0.02))
    nearby = NearbyAircraft(
        hex="abc",
        registration="N123AB",
        flight="N123AB",
        on_ground=True,
        ground_speed=5,
        seconds_since_seen=1,
    )
    engine = DetectionEngine(config, MockNearbyAircraftProvider([nearby]), publish)
    transmission_event = TranscriptEvent(
        event_id=str(uuid4()),
        timestamp=datetime.now(UTC),
        text="November one two three alpha bravo request taxi to Monterey Jet Center",
    )
    pending = await engine.process(transmission_event)
    assert isinstance(pending, PendingDestinationEvent)
    log_transmission_result(transmission_event, settings)
    assert "NOTIFICATION SENT" not in caplog.text
    await asyncio.sleep(0.04)
    assert "NOTIFICATION SENT" in caplog.text
    await engine.close()


class DeliverySocket:
    def __init__(self, fails: bool = False) -> None:
        self.fails = fails

    async def accept(self) -> None:
        return None

    async def send_text(self, _message: str) -> None:
        if self.fails:
            raise RuntimeError("closed")


@pytest.mark.asyncio
async def test_websocket_counts_drive_success_and_partial_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="mry_alert.operations")
    manager = WebSocketManager()
    await manager.connect(cast(WebSocket, DeliverySocket()))
    successful = await manager.broadcast(alert(event_id="success"))
    assert successful == DeliveryResult(1, 1, 0)
    log_notification_delivery(
        alert(event_id="success"), successful, LoggingConfig(), DeliveryLogTracker()
    )
    assert "NOTIFICATION SENT" in caplog.text

    caplog.clear()
    manager = WebSocketManager()
    await manager.connect(cast(WebSocket, DeliverySocket()))
    await manager.connect(cast(WebSocket, DeliverySocket(fails=True)))
    partial = await manager.broadcast(alert(event_id="partial"))
    assert partial == DeliveryResult(2, 1, 1)
    log_notification_delivery(
        alert(event_id="partial"), partial, LoggingConfig(), DeliveryLogTracker()
    )
    assert "NOTIFICATION DELIVERY PARTIAL" in caplog.text
    assert "Delivered:    1" in caplog.text
    assert "Failed:       1" in caplog.text

    caplog.clear()
    failed_event = alert(event_id="failed")
    log_notification_delivery(
        failed_event, DeliveryResult(1, 0, 1), LoggingConfig(), DeliveryLogTracker()
    )
    assert "NOTIFICATION DELIVERY FAILED" in caplog.text


def test_zero_clients_correction_and_duplicate_delivery_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="mry_alert.operations")
    settings = LoggingConfig()
    tracker = DeliveryLogTracker()
    original = alert(event_id="zero")
    log_notification_delivery(original, DeliveryResult(0, 0, 0), settings, tracker)
    assert "NOTIFICATION NOT DELIVERED" in caplog.text
    assert "No connected extension clients" in caplog.text

    caplog.clear()
    correction = alert(
        event_id="correction",
        event_type=AlertEventType.DESTINATION_CORRECTION,
        confirmation_status=ConfirmationStatus.CORRECTED,
        previous_destination="Monterey Jet Center",
        corrected_destination="Del Monte Aviation",
        destination="Del Monte Aviation",
        original_event_id="original",
    )
    log_notification_delivery(correction, DeliveryResult(1, 1, 0), settings, tracker)
    log_notification_delivery(correction, DeliveryResult(1, 1, 0), settings, tracker)
    messages = operational_messages(caplog)
    assert len(messages) == 1
    assert "CORRECTION NOTIFICATION SENT" in messages[0]
    assert "Original event ID:     original" in messages[0]


def test_pending_correction_has_explicit_cancellation_block(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="mry_alert.operations")
    pending = PendingDestinationEvent(
        **alert(
            confirmation_status=ConfirmationStatus.CORRECTED,
            event_type=AlertEventType.DESTINATION_CORRECTION,
            previous_destination="Monterey Jet Center",
            corrected_destination="Del Monte Aviation",
            destination="Del Monte Aviation",
        ).model_dump(exclude={"test"}),
        contact_key="N123AB",
    )
    log_pending_cancelled(pending, LoggingConfig())
    assert "PENDING ALERT CANCELLED" in caplog.text
    assert "State:                 CORRECTED" in caplog.text
    assert "Same-contact correction detected" in caplog.text
