from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import Lock

from mry_alert.config import LoggingConfig
from mry_alert.models import (
    AlertEvent,
    AlertEventType,
    PendingDestinationEvent,
    TranscriptEvent,
)

logger = logging.getLogger("mry_alert.operations")
THIN_RULE = "-" * 60
THICK_RULE = "=" * 60


@dataclass(frozen=True)
class DeliveryResult:
    connected: int
    delivered: int
    failed: int


@dataclass
class DeliveryLogTracker:
    _event_ids: set[str] = field(default_factory=set)
    _lock: Lock = field(default_factory=Lock)

    def first_log_for(self, event_id: str) -> bool:
        with self._lock:
            if event_id in self._event_ids:
                return False
            self._event_ids.add(event_id)
            return True


def _time(event: TranscriptEvent | AlertEvent | PendingDestinationEvent) -> str:
    return event.timestamp.astimezone().strftime("%H:%M:%S")


def _percent(value: float | None) -> str:
    return "unknown" if value is None else f"{round(value * 100)}%"


def _reason(reasons: list[str]) -> str:
    return "; ".join(reasons) if reasons else "No additional detector reason"


def log_transmission_result(event: TranscriptEvent, settings: LoggingConfig) -> None:
    if not settings.live_transcripts and not settings.detection_decisions:
        return
    heard = event.text.replace("\r", " ").replace("\n", " ").strip()
    if settings.format == "compact":
        lines: list[str] = []
        if settings.live_transcripts:
            lines.append(f'[{_time(event)}] HEARD "{heard}"')
        if settings.detection_decisions:
            subject = event.detected_callsign or "unknown contact"
            destination = event.destination_candidate or "none"
            lines.append(f"[{_time(event)}] {event.detection_decision} {subject} -> {destination}")
        logger.info("\n".join(lines))
        return

    lines = [THIN_RULE, f"ATC TRANSMISSION  {_time(event)}"]
    if settings.live_transcripts:
        lines.append(f'Heard:        "{heard}"')
        if settings.verbose_transcripts:
            lines.append(f'Normalized:   "{event.normalized_text}"')
        if settings.whisper_quality:
            lines.append(f"Decoder confidence: {event.whisper_quality or 'unknown'}")
            lines.append(
                "Decoder confidence reflects token-sequence confidence and does not guarantee "
                "transcription accuracy."
            )
            if event.average_log_probability is not None:
                lines.append(f"Average log probability: {event.average_log_probability:.2f}")
            if event.no_speech_probability is not None:
                lines.append(f"No-speech probability: {event.no_speech_probability:.2f}")
            if event.transcription_segment_count is not None:
                lines.append(f"Whisper segments: {event.transcription_segment_count}")
            if event.transcript_duration_seconds is not None:
                lines.append(f"Transcript duration: {event.transcript_duration_seconds:.1f}s")
        for recovery in event.normalization_reasons:
            lines.append(f'Intent raw:        "{heard}"')
            lines.append(f"Intent normalized: {event.normalized_text}")
            lines.append(f"Recovery reason:   {recovery}")
        if event.artifact_trimming_reason:
            lines.append(event.artifact_trimming_reason)
    if settings.detection_decisions:
        lines.extend(
            [
                f"Role:         {event.speaker_role} ({_percent(event.speaker_role_confidence)})",
                f"Callsign:     {event.detected_callsign or 'none'}",
                "Aircraft type: "
                f"{event.aircraft_type_name or event.aircraft_type_code or 'unknown'}",
                f"Intent:       {event.intent_category}",
                f"Route cues:   {', '.join(event.route_cues) or 'none'}",
                "Destination:  "
                f"{event.destination_candidate or 'none'} "
                f"({_percent(event.destination_candidate_confidence)})",
                f"Decision:     {event.detection_decision}",
                f"Reason:       {_reason(event.detection_reasons)}",
            ]
        )
        if settings.adsb_candidates and event.adsb_candidate_reasons:
            lines.append("ADS-B candidates:")
            lines.extend(
                f"  {index}. {candidate}"
                for index, candidate in enumerate(event.adsb_candidate_reasons, 1)
            )
            lines.append(f"Resolved as: {event.identified_registration or 'unresolved'}")
            lines.append(f"Identification source: {event.identification_source}")
    lines.append(THIN_RULE)
    logger.info("\n".join(lines))


def log_pending_cancelled(event: PendingDestinationEvent, settings: LoggingConfig) -> None:
    if not settings.detection_decisions:
        return
    aircraft = event.registration or event.spoken_callsign
    if settings.format == "compact":
        logger.info(
            "[%s] %s %s: %s -> %s",
            _time(event),
            event.confirmation_status.upper(),
            aircraft,
            event.previous_destination or event.destination,
            event.corrected_destination or "none",
        )
        return
    logger.info(
        "\n".join(
            [
                THICK_RULE,
                "PENDING ALERT CANCELLED",
                f"Aircraft:              {aircraft}",
                "Aircraft type:         "
                f"{event.aircraft_type_name or event.aircraft_type or 'unknown'}",
                f"Previous destination:  {event.previous_destination or event.destination}",
                f"Corrected destination: {event.corrected_destination or 'none'}",
                f"State:                 {event.confirmation_status.upper()}",
                "Reason:                Same-contact correction detected",
                THICK_RULE,
            ]
        )
    )


def log_notification_delivery(
    event: AlertEvent,
    result: DeliveryResult,
    settings: LoggingConfig,
    tracker: DeliveryLogTracker,
) -> None:
    if not settings.notification_delivery or not tracker.first_log_for(event.event_id):
        return
    aircraft = event.registration or event.spoken_callsign
    corrected = event.event_type in {
        AlertEventType.DESTINATION_CORRECTION,
        AlertEventType.DESTINATION_CANCELLED,
    }
    if result.connected == 0:
        title = "NOTIFICATION NOT DELIVERED"
    elif result.delivered == 0:
        title = "NOTIFICATION DELIVERY FAILED"
    elif result.failed:
        title = "NOTIFICATION DELIVERY PARTIAL"
    elif corrected:
        title = "CORRECTION NOTIFICATION SENT"
    else:
        title = "NOTIFICATION SENT"

    if settings.format == "compact":
        destination = event.corrected_destination or event.destination
        logger.info(
            "[%s] %s %s -> %s (%s) clients=%d delivered=%d failed=%d",
            _time(event),
            title,
            aircraft,
            destination,
            _percent(event.confidence),
            result.connected,
            result.delivered,
            result.failed,
        )
        return

    lines = [
        THICK_RULE,
        title,
        f"Aircraft:     {aircraft}",
        "Aircraft type: "
        f"{event.aircraft_type_name or event.aircraft_type or 'unknown'}",
    ]
    if corrected:
        lines.extend(
            [
                "Decision:              CORRECTED",
                f"Previous destination:  {event.previous_destination or event.destination}",
                f"Corrected destination: {event.corrected_destination or 'none'}",
                f"Original event ID:     {event.original_event_id or 'none'}",
            ]
        )
    else:
        lines.extend(
            [
                "Decision:     CONFIRMED",
                f"Callsign:     {event.spoken_callsign}",
                f"Destination:  {event.destination}",
                f"Identification source: {event.identification_source}",
                f"Confidence:   {_percent(event.confidence)}",
            ]
        )
    if result.connected == 0:
        lines.append("Reason:       No connected extension clients")
    elif result.failed == 0:
        client_label = "client" if result.delivered == 1 else "clients"
        lines.append(f"Extension:    Sent to {result.delivered} connected {client_label}")
    else:
        lines.extend(
            [
                f"Connected:    {result.connected}",
                f"Delivered:    {result.delivered}",
                f"Failed:       {result.failed}",
            ]
        )
    lines.append(f"Event ID:     {event.event_id}")
    lines.append(THICK_RULE)
    logger.info("\n".join(lines))
