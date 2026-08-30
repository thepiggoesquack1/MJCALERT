from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def utc_now() -> datetime:
    return datetime.now(UTC)


class RuntimeStatus:
    """Mutable, process-local observability state; it never affects detection decisions."""

    def __init__(self, speech_model: str, adsb_provider: str) -> None:
        self.started_at = utc_now()
        self.speech_model = speech_model
        self.speech_ready = False
        self.adsb_provider = adsb_provider
        self.adsb_ok = False
        self.adsb_tracked_aircraft = 0
        self.adsb_fresh_candidates = 0
        self.adsb_last_success_at: datetime | None = None
        self.adsb_error: str | None = None
        self.last_audio_received_at: datetime | None = None
        self.last_transcription_at: datetime | None = None
        self.last_transcript: str | None = None
        self.last_detected_callsign: str | None = None
        self.last_aircraft_type: str | None = None
        self.last_destination: str | None = None
        self.last_detector_decision: str | None = None
        self.last_adsb_correlation_at: datetime | None = None
        self.last_adsb_winner: str | None = None
        self.last_adsb_score: float | None = None
        self.last_adsb_margin: float | None = None
        self.recent_intent_result = "none"
        self.recent_intent_decision = "none"
        self.last_notification_at: datetime | None = None
        self.last_notification_success: bool | None = None
        self.last_notification_delivered = 0
        self.ntfy_last_attempt_at: datetime | None = None
        self.ntfy_last_success: bool | None = None
        self.ntfy_last_error: str | None = None
        self.classifier_enabled = False
        self.classifier_loaded = False
        self.classifier_model_version: str | None = None
        self.classifier_error: str | None = None
        self.last_classification_at: datetime | None = None
        self.last_classification: dict[str, object] | None = None
        self.dataset_collection_enabled = False
        self.pending_review_clips = 0
        self.training_clip_last_error: str | None = None
        self.training_clip_failures = 0

    @staticmethod
    def _iso(value: datetime | None) -> str | None:
        return value.isoformat() if value else None

    def snapshot(
        self,
        *,
        extension_clients: int,
        audio_status: str,
        input_mode: str,
        audio_recent_seconds: float = 15,
    ) -> dict[str, Any]:
        now = utc_now()
        audio_connected = audio_status not in {"idle", "error"}
        audio_recent = bool(
            self.last_audio_received_at
            and (now - self.last_audio_received_at).total_seconds() <= audio_recent_seconds
        )
        return {
            "server_running": True,
            "started_at": self._iso(self.started_at),
            "extension_clients": extension_clients,
            "audio_clients": int(audio_connected),
            "audio_connected": audio_connected,
            "audio_recently_received": audio_recent,
            "audio_status": audio_status,
            "input_mode": input_mode,
            "speech_model": self.speech_model,
            "speech_ready": self.speech_ready,
            "adsb_provider": self.adsb_provider,
            "adsb_ok": self.adsb_ok,
            "adsb_tracked_aircraft": self.adsb_tracked_aircraft,
            "adsb_fresh_candidates": self.adsb_fresh_candidates,
            "adsb_last_success_at": self._iso(self.adsb_last_success_at),
            "adsb_error": self.adsb_error,
            "last_audio_received_at": self._iso(self.last_audio_received_at),
            "last_transcription_at": self._iso(self.last_transcription_at),
            "last_transcript": self.last_transcript,
            "last_detected_callsign": self.last_detected_callsign,
            "last_aircraft_type": self.last_aircraft_type,
            "last_destination": self.last_destination,
            "last_detector_decision": self.last_detector_decision,
            "last_adsb_correlation_at": self._iso(self.last_adsb_correlation_at),
            "last_adsb_winner": self.last_adsb_winner,
            "last_adsb_score": self.last_adsb_score,
            "last_adsb_margin": self.last_adsb_margin,
            "recent_intent_result": self.recent_intent_result,
            "recent_intent_decision": self.recent_intent_decision,
            "last_notification_at": self._iso(self.last_notification_at),
            "last_notification_success": self.last_notification_success,
            "last_notification_delivered": self.last_notification_delivered,
            "ntfy_last_attempt_at": self._iso(self.ntfy_last_attempt_at),
            "ntfy_last_success": self.ntfy_last_success,
            "ntfy_last_error": self.ntfy_last_error,
            "classifier_enabled": self.classifier_enabled,
            "classifier_loaded": self.classifier_loaded,
            "classifier_model_version": self.classifier_model_version,
            "classifier_error": self.classifier_error,
            "last_classification_at": self._iso(self.last_classification_at),
            "last_classification": self.last_classification,
            "dataset_collection_enabled": self.dataset_collection_enabled,
            "pending_review_clips": self.pending_review_clips,
            "training_clip_last_error": self.training_clip_last_error,
            "training_clip_failures": self.training_clip_failures,
        }
