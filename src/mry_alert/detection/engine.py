from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import uuid4

from mry_alert.adsb.base import NearbyAircraftProvider
from mry_alert.adsb.traffic_filter import (
    TrafficDecision,
    TrafficFilter,
    TrafficFilteringProvider,
)
from mry_alert.audio_classifier.models import DestinationLabel, IntentLabel
from mry_alert.config import AppConfig
from mry_alert.detection.artifacts import trim_repetitive_tail
from mry_alert.detection.callsign import parse_callsign
from mry_alert.detection.context import ConversationContext, RadioContact
from mry_alert.detection.destination import (
    CorrectionEvidence,
    DestinationEvidence,
    ParsedDestination,
    detect_correction,
    detect_destination,
    parse_destination,
)
from mry_alert.detection.intent_normalizer import normalize_ground_intent
from mry_alert.detection.matcher import AircraftMatcher
from mry_alert.detection.normalizer import normalize_transcript
from mry_alert.detection.outbound import (
    OutboundDisposition,
    infer_outbound_departure,
)
from mry_alert.detection.speaker_role import infer_speaker_role
from mry_alert.models import (
    AlertEvent,
    AlertEventType,
    ConfirmationStatus,
    DestinationIntentCategory,
    DestinationState,
    DetectionDecision,
    IdentificationSource,
    MatchStatus,
    NearbyAircraft,
    PendingDestinationEvent,
    SpeakerRole,
    SpokenCallsign,
    TranscriptEvent,
)

DetectionEvent = AlertEvent | PendingDestinationEvent
EventPublisher = Callable[[DetectionEvent], Awaitable[None]]
logger = logging.getLogger(__name__)


@dataclass
class _PendingRecord:
    event: PendingDestinationEvent
    alert: AlertEvent
    spoken: SpokenCallsign | None = None
    raw_text: str = ""
    pilot_arrival_supported: bool = False
    strong_route: bool = False
    adsb_ground_context_observed: bool = False
    radio_identity_forms: list[str] | None = None
    task: asyncio.Task[None] | None = None


@dataclass
class _ConfirmedRecord:
    alert: AlertEvent
    detected_at: datetime


class DetectionEngine:
    def __init__(
        self,
        config: AppConfig,
        provider: NearbyAircraftProvider,
        publisher: EventPublisher | None = None,
        lifecycle_publisher: EventPublisher | None = None,
    ) -> None:
        self.config = config
        if isinstance(provider, TrafficFilteringProvider):
            self.provider = provider
            self.traffic_filter = provider.traffic_filter
        else:
            self.traffic_filter = TrafficFilter(config.traffic_filter)
            self.provider = TrafficFilteringProvider(provider, self.traffic_filter)
        self.context = ConversationContext(config.detection.context_window_seconds)
        self.matcher = AircraftMatcher(
            config.detection,
            config.adsb,
            config.adsb_matching,
            config.airport,
            config.adsb_tracking,
            config.adsb_geofences,
            config.adsb_scoring,
            config.adsb_decision,
        )
        self._publisher = publisher
        self._lifecycle_publisher = lifecycle_publisher
        self._duplicates: dict[tuple[str, str], datetime] = {}
        self._pending: dict[str, _PendingRecord] = {}
        self._confirmed: dict[str, _ConfirmedRecord] = {}
        self._state_lock = asyncio.Lock()

    def observe_adsb(self, nearby: list[NearbyAircraft]) -> None:
        """Add background observations to movement history used by later matches."""
        self.matcher.observe(nearby)

    @staticmethod
    def _callsigns_compatible(
        left: SpokenCallsign | None, right: SpokenCallsign | None
    ) -> bool:
        if left is None or right is None:
            return False
        left_value = left.full_registration or left.suffix or left.normalized_form
        right_value = right.full_registration or right.suffix or right.normalized_form
        left_value = left_value.replace("-", "").upper()
        right_value = right_value.replace("-", "").upper()
        if left_value.startswith("N"):
            left_value = left_value[1:]
        if right_value.startswith("N"):
            right_value = right_value[1:]
        return bool(
            min(len(left_value), len(right_value)) >= 3
            and (
                left_value == right_value
                or left_value.endswith(right_value)
                or right_value.endswith(left_value)
            )
        )

    @staticmethod
    def _display_callsign(callsign: SpokenCallsign | None) -> str | None:
        if callsign is None:
            return None
        if callsign.full_registration:
            return callsign.full_registration
        if callsign.aircraft_type_prefix and callsign.suffix:
            return f"{callsign.aircraft_type_prefix} {callsign.suffix}"
        return callsign.original_text.title()

    @staticmethod
    def _aircraft_metadata(aircraft: NearbyAircraft | None) -> dict[str, object]:
        if aircraft is None:
            return {}
        return {
            "operator_name": aircraft.operator_name,
            "aircraft_type": aircraft.aircraft_type,
            "aircraft_type_code": aircraft.aircraft_type_code,
            "aircraft_type_name": aircraft.aircraft_type_name,
            "manufacturer": aircraft.manufacturer,
            "model": aircraft.model,
            "aircraft_category": aircraft.aircraft_category,
            "aircraft_type_source": aircraft.aircraft_type_source,
            "aircraft_type_confidence": aircraft.aircraft_type_confidence,
        }

    def _record_decision(
        self,
        event: TranscriptEvent,
        decision: DetectionDecision,
        reasons: list[str],
        callsign: SpokenCallsign | None = None,
    ) -> None:
        event.detection_decision = decision
        event.detection_reasons = reasons
        if callsign is not None:
            event.detected_callsign = self._display_callsign(callsign)

    async def _emit(self, event: DetectionEvent) -> None:
        if self._publisher:
            await self._publisher(event)

    async def _emit_lifecycle(self, event: PendingDestinationEvent) -> None:
        if self._lifecycle_publisher:
            await self._lifecycle_publisher(event)

    def _expire_state(self, now: datetime) -> None:
        correction_window = timedelta(
            seconds=self.config.detection.destination_correction_window_seconds
        )
        self._confirmed = {
            key: record
            for key, record in self._confirmed.items()
            if now - record.detected_at <= correction_window
        }

    async def _confirm_after(self, contact_key: str, event_id: str) -> None:
        try:
            correlation_window = (
                self.config.adsb_decision.correlation_window_seconds
                if self.config.adsb_decision.correlation_window_seconds is not None
                else self.config.detection.destination_confirmation_delay_seconds
            )
            radio_delay = self.config.detection.destination_confirmation_delay_seconds
            await asyncio.sleep(radio_delay)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + max(0.0, correlation_window - radio_delay)
            while True:
                remaining = deadline - loop.time()
                final_attempt = remaining <= 0
                if await self._refresh_pending_identity(
                    contact_key, event_id, final_attempt=final_attempt
                ):
                    await self._confirm_pending(contact_key, event_id)
                    return
                async with self._state_lock:
                    still_pending = bool(
                        (record := self._pending.get(contact_key))
                        and record.event.event_id == event_id
                    )
                if not still_pending or final_attempt:
                    return
                await asyncio.sleep(
                    min(max(0.1, self.config.adsb.polling_interval_seconds), remaining)
                )
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Failed to confirm destination event %s", event_id)

    async def _refresh_pending_identity(
        self, contact_key: str, event_id: str, *, final_attempt: bool = True
    ) -> bool:
        async with self._state_lock:
            record = self._pending.get(contact_key)
            if not record or record.event.event_id != event_id:
                return False
            spoken, raw_text = record.spoken, record.raw_text
        try:
            nearby = await self.provider.nearby()
        except Exception:
            nearby = []
        match = self.matcher.match(spoken, nearby, raw_text)
        async with self._state_lock:
            record = self._pending.get(contact_key)
            if not record or record.event.event_id != event_id:
                return False
            record.adsb_ground_context_observed = bool(
                record.adsb_ground_context_observed
                or any(
                    "on_ground" in line or "recently_landed" in line
                    for line in match.candidate_scores
                )
            )
            if not record.pilot_arrival_supported:
                if not final_attempt:
                    logger.info(
                        "Arrival candidate %s is waiting for linked pilot evidence",
                        event_id,
                    )
                    return False
                self._pending.pop(contact_key, None)
                self.context.set_destination_state(
                    contact_key, DestinationState.CANCELLED, None, record.event.timestamp
                )
                expired = record.event.model_copy(
                    update={
                        "confirmation_status": ConfirmationStatus.EXPIRED,
                        "match_reasons": [
                            *record.event.match_reasons,
                            "controller-only candidate expired without linked pilot evidence",
                        ],
                    }
                )
                await self._emit_lifecycle(expired)
                logger.info(
                    "Controller-only arrival candidate expired without supporting "
                    "pilot evidence: %s",
                    event_id,
                )
                return False
            if (
                self.config.adsb_tracking.enabled
                and match.status != MatchStatus.CONFIRMED
                and not self.config.notifications.send_uncertain_alerts
            ):
                if record.alert.registration and record.alert.status == MatchStatus.CONFIRMED:
                    logger.info(
                        "Temporary ADS-B loss did not erase the previously confirmed "
                        "aircraft identity for event %s",
                        event_id,
                    )
                    return True
                identity_forms: list[str] = record.radio_identity_forms or []
                if (
                    self.config.notifications.allow_repeated_spoken_registration_fallback
                    and record.strong_route
                    and record.adsb_ground_context_observed
                    and spoken is not None
                    and spoken.full_registration is not None
                    and len(identity_forms)
                    >= self.config.notifications.minimum_spoken_registration_observations
                ):
                    update = {
                        "registration": spoken.full_registration,
                        "spoken_callsign": self._display_callsign(spoken)
                        or spoken.full_registration,
                        "confidence": min(record.alert.confidence, 0.86),
                        "status": MatchStatus.LIKELY,
                        "match_reasons": [
                            *record.alert.match_reasons,
                            "same radio identity was heard in multiple linked transmissions",
                            "nearby ADS-B confirmed KMRY ground traffic but did not "
                            "expose a matching registration",
                            "unrelated ADS-B tail was not guessed",
                        ],
                        "identification_source": IdentificationSource.SPOKEN_FULL_REGISTRATION,
                    }
                    record.alert = record.alert.model_copy(update=update)
                    record.event = record.event.model_copy(update=update)
                    logger.info(
                        "Repeated spoken registration resolved pending event %s as %s "
                        "without guessing another ADS-B tail",
                        event_id,
                        spoken.full_registration,
                    )
                    return True
                if not final_attempt:
                    logger.info(
                        "ADS-B identity unresolved; keeping event %s pending for a later "
                        "observation",
                        event_id,
                    )
                    return False
                if (
                    self.config.notifications.send_unidentified_arrival_alerts
                    and record.strong_route
                    and record.adsb_ground_context_observed
                ):
                    update = {
                        "registration": None,
                        "status": MatchStatus.UNRESOLVED,
                        "match_reasons": [
                            *record.alert.match_reasons,
                            *match.match_reasons,
                            "strong pilot arrival remained unresolved after the ADS-B "
                            "correlation window",
                            "alert sent without guessing an aircraft tail",
                        ],
                        "identification_source": IdentificationSource.UNRESOLVED,
                    }
                    record.alert = record.alert.model_copy(update=update)
                    record.event = record.event.model_copy(update=update)
                    logger.warning(
                        "Sending unidentified arrival alert for event %s after safe "
                        "identity resolution was exhausted",
                        event_id,
                    )
                    return True
                self._pending.pop(contact_key, None)
                self.context.set_destination_state(
                    contact_key, DestinationState.CANCELLED, None, record.event.timestamp
                )
                unresolved = record.event.model_copy(
                    update={
                        "confirmation_status": ConfirmationStatus.UNRESOLVED,
                        "match_reasons": [
                            *record.event.match_reasons,
                            *match.match_reasons,
                            "ADS-B correlation window ended without a clear identity",
                        ],
                    }
                )
                await self._emit_lifecycle(unresolved)
                logger.info(
                    "ADS-B correlation window ended without a clear identity; event withheld: %s",
                    "; ".join(match.match_reasons),
                )
                return False
            if match.registration:
                update = {
                    "registration": match.registration,
                    "confidence": min(record.alert.confidence, match.confidence),
                    "status": match.status,
                    "match_reasons": [*record.alert.match_reasons, *match.match_reasons],
                    "identification_source": match.identification_source,
                    "adsb_winning_score": match.winning_score,
                    "adsb_winning_margin": match.winning_margin,
                    "adsb_movement_state": match.movement_state,
                    **self._aircraft_metadata(match.aircraft),
                }
                record.alert = record.alert.model_copy(update=update)
                record.event = record.event.model_copy(update=update)
            return True

    async def _confirm_pending(self, contact_key: str, event_id: str) -> AlertEvent | None:
        async with self._state_lock:
            record = self._pending.get(contact_key)
            if not record or record.event.event_id != event_id:
                return None
            if not record.pilot_arrival_supported:
                logger.info(
                    "Pending event %s was not confirmed because linked pilot evidence is absent",
                    event_id,
                )
                return None
            self._pending.pop(contact_key, None)
            alert = record.alert.model_copy(
                update={"confirmation_status": ConfirmationStatus.CONFIRMED}
            )
            duplicate_key = (alert.registration or contact_key, alert.destination)
            previous = self._duplicates.get(duplicate_key)
            if previous and alert.timestamp - previous < timedelta(
                seconds=self.config.detection.duplicate_suppression_seconds
            ):
                self.context.set_destination_state(
                    contact_key, DestinationState.CANCELLED, None, alert.timestamp
                )
                await self._emit_lifecycle(
                    record.event.model_copy(
                        update={
                            "confirmation_status": ConfirmationStatus.CANCELLED,
                            "match_reasons": [
                                *record.event.match_reasons,
                                "duplicate_suppressed",
                            ],
                        }
                    )
                )
                return None
            self._duplicates[duplicate_key] = alert.timestamp
            self._confirmed[contact_key] = _ConfirmedRecord(alert, alert.timestamp)
            self.context.set_destination_state(
                contact_key, DestinationState.CONFIRMED, alert.destination, alert.timestamp
            )
            # Publish while holding the transition lock so a correction cannot
            # overtake this confirmation between its state change and emission.
            await self._emit(alert)
            return alert

    async def flush_pending(self) -> list[AlertEvent]:
        async with self._state_lock:
            pending = [
                (key, record.event.event_id, record.task) for key, record in self._pending.items()
            ]
            for _, _, task in pending:
                if task:
                    task.cancel()
        confirmed: list[AlertEvent] = []
        for contact_key, event_id, _ in pending:
            identity_ready = await self._refresh_pending_identity(
                contact_key, event_id, final_attempt=True
            )
            if identity_ready and (
                alert := await self._confirm_pending(contact_key, event_id)
            ):
                confirmed.append(alert)
        return confirmed

    async def close(self) -> None:
        async with self._state_lock:
            tasks = [record.task for record in self._pending.values() if record.task]
            self._pending.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _associated_correction_key(
        self, spoken: SpokenCallsign | None, timestamp: datetime
    ) -> str | None:
        async with self._state_lock:
            self._expire_state(timestamp)
            if spoken:
                key = spoken.normalized_form
                return key if key in self._pending or key in self._confirmed else None
            contact = self.context.unique_recent_contact(timestamp)
            if contact:
                key = contact.callsign.normalized_form
                return key if key in self._pending or key in self._confirmed else None
            candidates = set(self._pending) | set(self._confirmed)
            if self.context.contact_count(timestamp) == 0 and len(candidates) == 1:
                return next(iter(candidates))
            return None

    async def _handle_correction(
        self,
        contact_key: str,
        correction: CorrectionEvidence,
        event: TranscriptEvent,
    ) -> DetectionEvent | None:
        pending_update: PendingDestinationEvent | None = None
        correction_alert: AlertEvent | None = None
        async with self._state_lock:
            pending = self._pending.pop(contact_key, None)
            if pending:
                if pending.task:
                    pending.task.cancel()
                confirmation = (
                    ConfirmationStatus.CORRECTED
                    if correction.corrected_destination
                    else ConfirmationStatus.CANCELLED
                )
                event_type = (
                    AlertEventType.DESTINATION_CORRECTION
                    if correction.corrected_destination
                    else AlertEventType.DESTINATION_CANCELLED
                )
                pending_update = pending.event.model_copy(
                    update={
                        "timestamp": event.timestamp,
                        "event_type": event_type,
                        "previous_destination": pending.event.destination,
                        "corrected_destination": correction.corrected_destination,
                        "confirmation_status": confirmation,
                        "transcript_excerpt": event.text,
                        "match_reasons": [*pending.event.match_reasons, *correction.reasons],
                        "speaker_role": event.speaker_role,
                        "speaker_role_confidence": event.speaker_role_confidence,
                        "speaker_role_reasons": event.speaker_role_reasons,
                    }
                )
                state = (
                    DestinationState.CORRECTED
                    if correction.corrected_destination
                    else DestinationState.CANCELLED
                )
                self.context.set_destination_state(
                    contact_key, state, correction.corrected_destination, event.timestamp
                )
            else:
                confirmed = self._confirmed.pop(contact_key, None)
                if confirmed:
                    original = confirmed.alert
                    confirmation = (
                        ConfirmationStatus.CORRECTED
                        if correction.corrected_destination
                        else ConfirmationStatus.CANCELLED
                    )
                    event_type = (
                        AlertEventType.DESTINATION_CORRECTION
                        if correction.corrected_destination
                        else AlertEventType.DESTINATION_CANCELLED
                    )
                    correction_alert = AlertEvent(
                        event_id=str(uuid4()),
                        timestamp=event.timestamp,
                        destination=correction.corrected_destination or original.destination,
                        registration=original.registration,
                        spoken_callsign=original.spoken_callsign,
                        aircraft_type=original.aircraft_type,
                        operator_name=original.operator_name,
                        aircraft_type_code=original.aircraft_type_code,
                        aircraft_type_name=original.aircraft_type_name,
                        manufacturer=original.manufacturer,
                        model=original.model,
                        aircraft_category=original.aircraft_category,
                        aircraft_type_source=original.aircraft_type_source,
                        aircraft_type_confidence=original.aircraft_type_confidence,
                        confidence=min(original.confidence, correction.confidence),
                        status=original.status,
                        transcript_excerpt=event.text,
                        match_reasons=[*original.match_reasons, *correction.reasons],
                        alternative_registrations=original.alternative_registrations,
                        event_type=event_type,
                        previous_destination=original.destination,
                        corrected_destination=correction.corrected_destination,
                        original_event_id=original.event_id,
                        confirmation_status=confirmation,
                        speaker_role=event.speaker_role,
                        speaker_role_confidence=event.speaker_role_confidence,
                        speaker_role_reasons=event.speaker_role_reasons,
                        identification_source=original.identification_source,
                        adsb_winning_score=original.adsb_winning_score,
                        adsb_winning_margin=original.adsb_winning_margin,
                        adsb_movement_state=original.adsb_movement_state,
                        intent=event.intent_category,
                        classifier_confidence=(
                            max(
                                event.audio_intent.destination_confidence,
                                event.audio_intent.intent_confidence,
                                event.audio_intent.correction_confidence,
                            )
                            if event.audio_intent
                            else original.classifier_confidence
                        ),
                        whisper_confidence=event.transcription_confidence,
                        decoder_confidence=event.whisper_quality,
                    )
                    state = (
                        DestinationState.CORRECTED
                        if correction.corrected_destination
                        else DestinationState.CANCELLED
                    )
                    self.context.set_destination_state(
                        contact_key, state, correction.corrected_destination, event.timestamp
                    )
        result = pending_update or correction_alert
        if result:
            await self._emit(result)
        return result

    async def _build_arrival(
        self,
        event: TranscriptEvent,
        spoken: SpokenCallsign | None,
        destination_reasons: list[str],
        destination_confidence: float,
        allow_unresolved_identity: bool = False,
    ) -> tuple[
        AlertEvent | None,
        MatchStatus,
        list[str],
        OutboundDisposition,
    ]:
        try:
            nearby = await self.provider.nearby()
        except Exception:
            nearby = []
        match = self.matcher.match(spoken, nearby, event.normalized_text)
        if self.config.adsb_tracking.enabled:
            logger.info(
                "ADS-B CORRELATION\nSpeech destination: %s\nSpeech callsign clue: %s\n"
                "Correlation window: %.1f seconds\nCandidates considered:\n%s\n"
                "Decision: %s\nWinning score: %s\nSecond-best score: %s\nMargin: %s\nReason: %s",
                event.destination_candidate or self.config.destination.canonical_name,
                self._display_callsign(spoken) or "none",
                self.config.adsb_decision.correlation_window_seconds
                if self.config.adsb_decision.correlation_window_seconds is not None
                else self.config.detection.destination_confirmation_delay_seconds,
                "\n".join(match.candidate_scores) or "none",
                match.adsb_decision.value if match.adsb_decision else match.status.value,
                match.winning_score,
                match.second_best_score,
                match.winning_margin,
                "; ".join(match.match_reasons),
            )
        event.identification_source = match.identification_source
        event.identified_registration = match.registration
        event.adsb_candidate_reasons = match.candidate_scores
        event.adsb_decision = match.adsb_decision
        event.adsb_winning_score = match.winning_score
        event.adsb_winning_margin = match.winning_margin
        event.adsb_movement_state = match.movement_state
        if match.aircraft:
            event.identified_operator = match.aircraft.operator_name
            event.aircraft_type_code = match.aircraft.aircraft_type_code
            event.aircraft_type_name = match.aircraft.aircraft_type_name
            event.manufacturer = match.aircraft.manufacturer
            event.model = match.aircraft.model
            event.aircraft_category = match.aircraft.aircraft_category
            event.aircraft_type_source = match.aircraft.aircraft_type_source
            event.aircraft_type_confidence = match.aircraft.aircraft_type_confidence
        if (
            event.audio_intent is not None
            and self.config.audio_classifier.enabled
            and self.config.audio_classifier.require_adsb_for_notification
            and (
                not match.registration
                or match.status in {MatchStatus.AMBIGUOUS, MatchStatus.UNRESOLVED}
            )
        ):
            return (
                None,
                match.status,
                [
                    *match.match_reasons,
                    "audio-classifier policy requires a resolved ADS-B aircraft",
                ],
                OutboundDisposition.ARRIVAL_ELIGIBLE,
            )
        if (
            self.config.adsb_tracking.enabled
            and match.status in {MatchStatus.AMBIGUOUS, MatchStatus.UNRESOLVED}
            and not self.config.notifications.send_uncertain_alerts
            and not allow_unresolved_identity
        ):
            return (
                None,
                match.status,
                match.match_reasons,
                OutboundDisposition.ARRIVAL_ELIGIBLE,
            )
        outbound = infer_outbound_departure(event.normalized_text, match)
        if outbound.disposition != OutboundDisposition.ARRIVAL_ELIGIBLE:
            logger.info(
                "OUTBOUND MOVEMENT CHECK\nAircraft: %s\nDecision: %s\nReason:\n%s",
                match.registration or self._display_callsign(spoken) or "unresolved",
                outbound.disposition.value,
                "\n".join(outbound.reasons),
            )
            return (
                None,
                MatchStatus.UNRESOLVED,
                [outbound.disposition.value, *outbound.reasons],
                outbound.disposition,
            )
        confidence = (
            destination_confidence
            if allow_unresolved_identity
            and match.status in {MatchStatus.AMBIGUOUS, MatchStatus.UNRESOLVED}
            else min(destination_confidence, match.confidence)
        )
        if (
            match.status in {MatchStatus.AMBIGUOUS, MatchStatus.UNRESOLVED}
            and not self.config.detection.notify_on_unresolved
            and not allow_unresolved_identity
        ):
            return (
                None,
                match.status,
                match.match_reasons,
                OutboundDisposition.ARRIVAL_ELIGIBLE,
            )
        unique_ground = (
            match.identification_source.value == "unique_ground_candidate"
            and match.confidence >= self.config.adsb_matching.unique_candidate_minimum_confidence
        )
        if (
            confidence < self.config.detection.alert_threshold
            and match.status not in {MatchStatus.AMBIGUOUS, MatchStatus.UNRESOLVED}
            and not unique_ground
        ):
            return (
                None,
                match.status,
                [
                    *match.match_reasons,
                    "combined confidence was below the configured alert threshold",
                ],
                OutboundDisposition.ARRIVAL_ELIGIBLE,
            )
        registration = match.registration
        alternatives = [a.registration for a in match.alternative_candidates if a.registration]
        alert = AlertEvent(
            event_id=str(uuid4()),
            timestamp=event.timestamp,
            destination=self.config.destination.canonical_name,
            registration=registration,
            spoken_callsign=self._display_callsign(spoken) or "Unresolved aircraft",
            confidence=confidence,
            status=match.status,
            transcript_excerpt=event.text,
            match_reasons=destination_reasons + match.match_reasons,
            alternative_registrations=alternatives,
            confirmation_status=ConfirmationStatus.PENDING,
            speaker_role=event.speaker_role,
            speaker_role_confidence=event.speaker_role_confidence,
            speaker_role_reasons=event.speaker_role_reasons,
            identification_source=match.identification_source,
            adsb_winning_score=match.winning_score,
            adsb_winning_margin=match.winning_margin,
            adsb_movement_state=match.movement_state,
            intent=event.intent_category,
            classifier_confidence=(
                max(
                    event.audio_intent.destination_confidence,
                    event.audio_intent.intent_confidence,
                    event.audio_intent.correction_confidence,
                )
                if event.audio_intent
                else None
            ),
            whisper_confidence=event.transcription_confidence,
            decoder_confidence=event.whisper_quality,
        ).model_copy(update=self._aircraft_metadata(match.aircraft))
        return (
            alert,
            match.status,
            match.match_reasons,
            OutboundDisposition.ARRIVAL_ELIGIBLE,
        )

    async def _create_pending(
        self,
        contact_key: str,
        alert: AlertEvent,
        spoken: SpokenCallsign | None = None,
        raw_text: str = "",
        *,
        pilot_arrival_supported: bool,
        strong_route: bool,
        adsb_ground_context_observed: bool,
    ) -> tuple[PendingDestinationEvent | None, str | None]:
        pending_event = PendingDestinationEvent(
            **alert.model_dump(exclude={"test"}),
            contact_key=contact_key,
        )
        record = _PendingRecord(
            pending_event,
            alert,
            spoken,
            raw_text,
            pilot_arrival_supported=pilot_arrival_supported,
            strong_route=strong_route,
            adsb_ground_context_observed=adsb_ground_context_observed,
            radio_identity_forms=[spoken.normalized_form] if spoken else [],
        )
        async with self._state_lock:
            if alert.registration and not contact_key.startswith("route:"):
                correlation_seconds = (
                    self.config.adsb_decision.correlation_window_seconds
                    if self.config.adsb_decision.correlation_window_seconds is not None
                    else self.config.detection.context_window_seconds
                )
                anonymous = [
                    (key, candidate)
                    for key, candidate in self._pending.items()
                    if key.startswith("route:")
                    and candidate.alert.destination == alert.destination
                    and alert.timestamp - candidate.alert.timestamp
                    <= timedelta(seconds=correlation_seconds)
                ]
                if len(anonymous) == 1:
                    anonymous_key, anonymous_record = anonymous[0]
                    self._pending.pop(anonymous_key, None)
                    if anonymous_record.task:
                        anonymous_record.task.cancel()
                    logger.info(
                        "Later radio/ADS-B evidence resolved anonymous arrival candidate %s as %s",
                        anonymous_record.event.event_id,
                        alert.registration,
                    )
            previous = self._pending.get(contact_key)
            if previous or contact_key in self._confirmed:
                return None, "suppressed_active_contact"
            duplicate_key = (alert.registration or contact_key, alert.destination)
            last_confirmation = self._duplicates.get(duplicate_key)
            if last_confirmation and alert.timestamp - last_confirmation < timedelta(
                seconds=self.config.detection.duplicate_suppression_seconds
            ):
                return None, "duplicate_suppressed"
            self._pending[contact_key] = record
            self.context.set_destination_state(
                contact_key, DestinationState.PENDING, alert.destination, alert.timestamp
            )
            try:
                await self._emit(pending_event)
            except Exception:
                self._pending.pop(contact_key, None)
                self.context.set_destination_state(
                    contact_key, DestinationState.CANCELLED, None, alert.timestamp
                )
                raise
            record.task = asyncio.create_task(
                self._confirm_after(contact_key, pending_event.event_id),
                name=f"confirm-destination-{contact_key}",
            )
        return pending_event, None

    async def _link_pending_followup(
        self,
        event: TranscriptEvent,
        spoken: SpokenCallsign | None,
        *,
        pilot_arrival_supported: bool,
        destination_present: bool,
    ) -> tuple[bool, DetectionEvent | None]:
        """Attach a later callsign/readback to one safe, recent pending arrival."""
        if spoken is None:
            return False, None
        correlation_seconds = (
            self.config.adsb_decision.correlation_window_seconds
            if self.config.adsb_decision.correlation_window_seconds is not None
            else self.config.detection.context_window_seconds
        )
        async with self._state_lock:
            compatible = [
                (key, record)
                for key, record in self._pending.items()
                if not key.startswith("route:")
                and self._callsigns_compatible(record.spoken, spoken)
                and event.timestamp - record.event.timestamp
                <= timedelta(seconds=correlation_seconds)
            ]
            candidates = compatible
            if not candidates:
                anonymous = [
                    (key, record)
                    for key, record in self._pending.items()
                    if key.startswith("route:")
                    and event.timestamp - record.event.timestamp
                    <= timedelta(seconds=self.config.detection.pending_followup_link_seconds)
                ]
                candidates = anonymous
            if len(candidates) != 1:
                return False, None
            old_key, record = candidates[0]
            event_id = record.event.event_id
            previously_resolved = bool(
                record.alert.registration and record.alert.status == MatchStatus.CONFIRMED
            )

        try:
            nearby = await self.provider.nearby()
        except Exception:
            nearby = []
        match = self.matcher.match(spoken, nearby, event.normalized_text)

        async with self._state_lock:
            refreshed_record = self._pending.get(old_key)
            if refreshed_record is None or refreshed_record.event.event_id != event_id:
                return False, None
            record = refreshed_record
            new_key = spoken.normalized_form
            if new_key != old_key and new_key in self._pending:
                logger.info(
                    "Later callsign %s was not linked because another pending contact "
                    "owns that key",
                    new_key,
                )
                return False, None
            record.spoken = (
                spoken
                if spoken.full_registration or record.spoken is None
                else record.spoken
            )
            record.raw_text = f"{record.raw_text} {event.normalized_text}".strip()
            record.pilot_arrival_supported = bool(
                record.pilot_arrival_supported or pilot_arrival_supported
            )
            if record.radio_identity_forms is None:
                record.radio_identity_forms = []
            record.radio_identity_forms.append(spoken.normalized_form)
            record.adsb_ground_context_observed = bool(
                record.adsb_ground_context_observed
                or any(
                    "on_ground" in line or "recently_landed" in line
                    for line in match.candidate_scores
                )
            )
            linked_reasons = [
                *record.alert.match_reasons,
                "later callsign/readback linked to the pending arrival by phraseology "
                "and turn-taking",
            ]
            update: dict[str, object] = {
                "spoken_callsign": self._display_callsign(record.spoken) or "Unresolved aircraft",
                "speaker_role": event.speaker_role,
                "speaker_role_confidence": event.speaker_role_confidence,
                "speaker_role_reasons": event.speaker_role_reasons,
                "match_reasons": linked_reasons,
            }
            if match.registration and match.status == MatchStatus.CONFIRMED:
                update.update(
                    {
                        "registration": match.registration,
                        "confidence": min(record.alert.confidence, match.confidence),
                        "status": match.status,
                        "identification_source": match.identification_source,
                        "adsb_winning_score": match.winning_score,
                        "adsb_winning_margin": match.winning_margin,
                        "adsb_movement_state": match.movement_state,
                        "match_reasons": [*linked_reasons, *match.match_reasons],
                        **self._aircraft_metadata(match.aircraft),
                    }
                )
            record.alert = record.alert.model_copy(update=update)
            record.event = record.event.model_copy(update=update)
            if new_key != old_key:
                self._pending.pop(old_key)
                if record.task:
                    record.task.cancel()
                self._pending[new_key] = record
                self.context.set_destination_state(
                    new_key,
                    DestinationState.PENDING,
                    record.alert.destination,
                    event.timestamp,
                )
                record.task = asyncio.create_task(
                    self._confirm_after(new_key, event_id),
                    name=f"confirm-destination-{new_key}",
                )
            linked_key = new_key
            updated_pending = record.event

        if (
            record.pilot_arrival_supported
            and not previously_resolved
            and await self._refresh_pending_identity(
                linked_key, event_id, final_attempt=False
            )
        ):
            confirmed = await self._confirm_pending(linked_key, event_id)
            if confirmed is not None:
                return True, confirmed
        if not destination_present:
            await self._emit(updated_pending)
            return True, updated_pending
        return True, None

    async def process(self, event: TranscriptEvent) -> DetectionEvent | None:
        trimmed = trim_repetitive_tail(event.text)
        event.artifact_trimming_reason = trimmed.reason
        base_normalized = normalize_transcript(trimmed.text)
        authorized_ground_source = event.source == self.config.liveatc.source_label
        intent_normalization = normalize_ground_intent(
            base_normalized, authorized_ground_source=authorized_ground_source
        )
        event.normalized_text = intent_normalization.text
        event.normalization_reasons = intent_normalization.reasons
        traffic_decision = self.traffic_filter.evaluate_transcript(event.normalized_text)
        event.traffic_filter_decision = traffic_decision.decision.value
        event.traffic_filter_reasons = [traffic_decision.reason]
        if traffic_decision.decision == TrafficDecision.IGNORED:
            self._record_decision(
                event,
                DetectionDecision.IGNORED,
                [traffic_decision.reason],
            )
            return None
        parsed_destination = parse_destination(event.normalized_text, self.config.destinations)
        classification = event.audio_intent
        classifier_primary = bool(
            classification
            and self.config.audio_classifier.enabled
            and self.config.decision_fusion.classifier_primary
        )
        if classifier_primary and classification:
            logger.info(
                "AUDIO CLASSIFIER\nDestination: %s\nDestination confidence: %.3f\n"
                "Intent: %s\nIntent confidence: %.3f\nCorrection: %s\n"
                "Noise probability: %.3f\nWhisper transcript: %r",
                classification.destination.value,
                classification.destination_confidence,
                classification.intent.value,
                classification.intent_confidence,
                classification.correction,
                classification.noise_confidence,
                event.text,
            )
            if (
                classification.noise_confidence
                >= self.config.audio_classifier.noise_rejection_threshold
                or classification.intent == IntentLabel.NOISE
            ):
                event.fusion_decision = (
                    "low-quality follow-up ignored; existing pending evidence preserved"
                )
                self._record_decision(
                    event,
                    DetectionDecision.IGNORED,
                    ["audio classifier rejected noise or unintelligible audio"],
                )
                return None
            destination_name = {
                DestinationLabel.MONTEREY_JET_CENTER: self.config.destination.canonical_name,
                DestinationLabel.DEL_MONTE_AVIATION: "Del Monte Aviation",
            }.get(classification.destination)
            if (
                destination_name
                and classification.destination_confidence
                >= self.config.audio_classifier.confidence_threshold
            ):
                parsed_destination = ParsedDestination(
                    destination_name,
                    classification.destination.value,
                    classification.destination_confidence,
                    0,
                    0,
                )
        spoken = parse_callsign(event.normalized_text)
        inherited: SpokenCallsign | None = None
        ambiguous_context = False
        if spoken is None and parsed_destination is not None:
            inherited, ambiguous_context = self.context.parking_response_contact(event.timestamp)
        role = infer_speaker_role(
            event.normalized_text,
            spoken,
            responding_to_destination_prompt=inherited is not None,
        )
        event.speaker_role = role.role
        event.speaker_role_confidence = role.confidence
        event.speaker_role_reasons = role.reasons
        self.context.observe(
            spoken,
            event.normalized_text,
            event.timestamp,
            role.role,
            role.confidence,
        )
        associated_spoken = spoken or inherited
        event.detected_callsign = self._display_callsign(associated_spoken)
        if parsed_destination:
            event.destination_candidate = parsed_destination.canonical_name
            event.destination_candidate_confidence = parsed_destination.confidence

        phraseology_pilot_support = bool(
            (event.speaker_role == SpeakerRole.PILOT and event.speaker_role_confidence >= 0.55)
            or self.context.has_pilot_arrival_evidence(associated_spoken, event.timestamp)
        )
        linked, linked_event = await self._link_pending_followup(
            event,
            associated_spoken,
            pilot_arrival_supported=phraseology_pilot_support,
            destination_present=parsed_destination is not None,
        )
        if linked and isinstance(linked_event, AlertEvent):
            self._record_decision(
                event,
                DetectionDecision.CONFIRMED,
                ["later radio/ADS-B evidence safely resolved a pending arrival"],
                associated_spoken,
            )
            return linked_event
        if linked and isinstance(linked_event, PendingDestinationEvent):
            self._record_decision(
                event,
                DetectionDecision.PENDING,
                ["callsign/readback enriched an existing pending arrival"],
                associated_spoken,
            )
            return linked_event
        if (
            linked
            and parsed_destination is not None
            and parsed_destination.canonical_name == self.config.destination.canonical_name
        ):
            self._record_decision(
                event,
                DetectionDecision.PENDING,
                ["additional evidence merged into an existing pending arrival"],
                associated_spoken,
            )
            return None

        correction_key = await self._associated_correction_key(associated_spoken, event.timestamp)
        if correction_key:
            previous = self.config.destination.canonical_name
            if (
                classifier_primary
                and classification
                and classification.correction_confidence
                >= self.config.audio_classifier.correction_threshold
            ):
                corrected = (
                    parsed_destination.canonical_name
                    if parsed_destination and parsed_destination.canonical_name != previous
                    else None
                )
                correction = CorrectionEvidence(
                    True,
                    classification.correction_confidence,
                    corrected,
                    ["audio classifier detected a correction or destination change"],
                )
            else:
                correction = detect_correction(event.normalized_text, parsed_destination, previous)
            if (
                not correction.detected
                and parsed_destination
                and parsed_destination.canonical_name != previous
                and associated_spoken
            ):
                correction = CorrectionEvidence(
                    True,
                    parsed_destination.confidence,
                    parsed_destination.canonical_name,
                    ["newer explicit destination for the same active contact"],
                )
            if correction.detected:
                result = await self._handle_correction(correction_key, correction, event)
                if result:
                    decision = (
                        DetectionDecision.CORRECTED
                        if correction.corrected_destination
                        else DetectionDecision.CANCELLED
                    )
                    self._record_decision(event, decision, correction.reasons, associated_spoken)
                    return result

        responding = inherited is not None or ambiguous_context
        evidence = detect_destination(
            event.normalized_text,
            self.config.destination,
            responding,
            authorized_ground_source,
            self.config.detection.route_intent_threshold,
            self.config.detection.fuzzy_intent_matching,
            self.config.detection.fuzzy_intent_minimum_score,
            self.config.intent_detection,
        )
        contextual_route_cues = self.context.route_cues(associated_spoken, event.timestamp)
        if (
            not evidence.detected
            and evidence.exact_destination
            and contextual_route_cues
            and self.config.intent_detection.allow_contextual_destination_inference
        ):
            evidence = DestinationEvidence(
                True,
                max(
                    self.config.intent_detection.destination_phrase_threshold,
                    self.config.intent_detection.weak_phrase_with_strong_route_threshold,
                ),
                [
                    *evidence.reasons,
                    "destination phrase agreed with recent linked route context",
                ],
                evidence.exact_destination,
                True,
                DestinationIntentCategory.GROUND_ROUTE_TO_DESTINATION,
                list(dict.fromkeys([*evidence.route_cues, *contextual_route_cues])),
            )
        if classifier_primary and classification:
            strong_destination = (
                classification.destination == DestinationLabel.MONTEREY_JET_CENTER
                and classification.destination_confidence
                >= self.config.audio_classifier.confidence_threshold
            )
            strong_intent = (
                classification.intent
                in {
                    IntentLabel.TAXI_OR_ROUTE,
                    IntentLabel.PARKING_STATEMENT,
                    IntentLabel.PARKING_PROMPT_RESPONSE,
                }
                and classification.intent_confidence
                >= self.config.audio_classifier.confidence_threshold
            )
            if strong_destination and strong_intent:
                category = {
                    IntentLabel.TAXI_OR_ROUTE: (
                        DestinationIntentCategory.GROUND_ROUTE_TO_DESTINATION
                    ),
                    IntentLabel.PARKING_STATEMENT: (
                        DestinationIntentCategory.EXPLICIT_PARKING_STATEMENT
                    ),
                    IntentLabel.PARKING_PROMPT_RESPONSE: (
                        DestinationIntentCategory.PARKING_PROMPT_RESPONSE
                    ),
                }[classification.intent]
                evidence = DestinationEvidence(
                    True,
                    min(
                        classification.destination_confidence,
                        classification.intent_confidence,
                    ),
                    [
                        "audio classifier identified Monterey Jet Center",
                        f"audio classifier intent {classification.intent.value}",
                    ],
                    True,
                    classification.intent == IntentLabel.TAXI_OR_ROUTE,
                    category,
                    ["audio classifier"],
                )
                event.fusion_decision = (
                    "classifier accepted as primary; Whisper disagreement ignored"
                )
            elif not (
                self.config.audio_classifier.allow_whisper_fallback
                and self.config.decision_fusion.allow_whisper_only_alerts
            ):
                event.fusion_decision = (
                    "classifier evidence insufficient; Whisper-only alert disabled"
                )
                evidence = DestinationEvidence(
                    False,
                    classification.destination_confidence,
                    ["classifier evidence did not meet destination and intent thresholds"],
                )
        event.intent_category = evidence.intent_category
        event.route_cues = evidence.route_cues
        sensitivity = self.config.detection.alert_sensitivity
        never_miss_mention = bool(
            sensitivity == "never_miss"
            and parsed_destination is not None
            and parsed_destination.canonical_name == self.config.destination.canonical_name
            and authorized_ground_source
            and not ambiguous_context
        )
        if never_miss_mention and not evidence.detected:
            assert parsed_destination is not None
            evidence = DestinationEvidence(
                True,
                max(parsed_destination.confidence, 0.75),
                [
                    *evidence.reasons,
                    "never-miss mode accepted an exact Jet Center mention for ADS-B review",
                ],
                True,
                False,
                DestinationIntentCategory.WEAK_DESTINATION_MENTION,
                evidence.route_cues,
            )
            event.intent_category = evidence.intent_category
        if parsed_destination and (
            parsed_destination.canonical_name != self.config.destination.canonical_name
        ):
            self._record_decision(
                event,
                DetectionDecision.IGNORED,
                ["explicit destination is not Monterey Jet Center"],
                associated_spoken,
            )
            return None
        if not evidence.detected or ambiguous_context:
            if ambiguous_context:
                reasons = ["multiple active contacts make the short reply ambiguous"]
                decision = DetectionDecision.AMBIGUOUS
            elif parsed_destination:
                reasons = ["Monterey Jet Center mention lacked taxi, parking, or prompt context"]
                decision = DetectionDecision.UNRESOLVED
            else:
                reasons = ["No Monterey Jet Center destination detected"]
                decision = DetectionDecision.IGNORED
            self._record_decision(event, decision, reasons, associated_spoken)
            return None
        if event.destination_candidate is None:
            event.destination_candidate = self.config.destination.canonical_name
            event.destination_candidate_confidence = evidence.confidence
        strong_route = (
            evidence.intent_category
            in {
                DestinationIntentCategory.EXPLICIT_TAXI_REQUEST,
                DestinationIntentCategory.EXPLICIT_PARKING_STATEMENT,
                DestinationIntentCategory.GROUND_ROUTE_TO_DESTINATION,
            }
            and evidence.confidence >= self.config.detection.strong_destination_threshold
            and authorized_ground_source
        )
        if (
            evidence.intent_category == DestinationIntentCategory.GROUND_ROUTE_TO_DESTINATION
            and evidence.confidence >= self.config.intent_detection.destination_phrase_threshold
            and authorized_ground_source
        ):
            strong_route = True
        if never_miss_mention:
            strong_route = True
        pilot_arrival_support = phraseology_pilot_support
        controller_only_route = bool(
            event.speaker_role == SpeakerRole.CONTROLLER
            and event.speaker_role_confidence >= 0.65
            and not pilot_arrival_support
        )

        contact: RadioContact | None = None
        if associated_spoken:
            contact = self.context.contact(associated_spoken, event.timestamp)
        if contact is None and not strong_route:
            contact = self.context.unique_recent_contact(event.timestamp)
        if contact is None and not strong_route:
            self._record_decision(
                event,
                DetectionDecision.AMBIGUOUS,
                ["no unambiguous active aircraft contact could be linked"],
                associated_spoken,
            )
            return None

        arrival, match_status, match_reasons, outbound_disposition = await self._build_arrival(
            event,
            contact.callsign if contact else None,
            evidence.reasons,
            evidence.confidence,
            allow_unresolved_identity=strong_route,
        )
        if arrival is None:
            if outbound_disposition == OutboundDisposition.OUTBOUND_DEPARTURE_FILTERED:
                decision = DetectionDecision.IGNORED
            else:
                decision = {
                    MatchStatus.AMBIGUOUS: DetectionDecision.AMBIGUOUS,
                    MatchStatus.UNRESOLVED: DetectionDecision.UNRESOLVED,
                }.get(match_status, DetectionDecision.IGNORED)
            self._record_decision(
                event,
                decision,
                match_reasons or ["aircraft match did not qualify for an alert"],
                contact.callsign if contact else None,
            )
            return None
        balanced_controller_support = bool(
            sensitivity == "balanced"
            and controller_only_route
            and strong_route
            and arrival.status == MatchStatus.CONFIRMED
            and arrival.registration is not None
            and arrival.adsb_movement_state == "moving_toward_jet_center"
        )
        never_miss_support = bool(
            sensitivity == "never_miss"
            and never_miss_mention
            and event.adsb_candidate_reasons
            and arrival.adsb_movement_state
            not in {
                "parked_at_jet_center",
                "moving_away_from_jet_center",
                "departing",
                "stale",
            }
        )
        sensitivity_support = balanced_controller_support or never_miss_support
        if sensitivity_support:
            reason = (
                "balanced mode accepted controller routing with a confirmed aircraft "
                "moving toward Monterey Jet Center"
                if balanced_controller_support
                else "never-miss mode accepted a plausible ground aircraft for an exact "
                "Jet Center mention"
            )
            arrival = arrival.model_copy(
                update={
                    "status": MatchStatus.LIKELY,
                    "match_reasons": [*arrival.match_reasons, reason],
                }
            )
        contact_key = contact.callsign.normalized_form if contact else f"route:{event.event_id}"
        pending, suppression_reason = await self._create_pending(
            contact_key,
            arrival,
            contact.callsign if contact else associated_spoken,
            event.normalized_text,
            pilot_arrival_supported=pilot_arrival_support or sensitivity_support,
            strong_route=strong_route,
            adsb_ground_context_observed=any(
                "on_ground" in line or "recently_landed" in line
                for line in event.adsb_candidate_reasons
            ),
        )
        if pending is None:
            self._record_decision(
                event,
                DetectionDecision.IGNORED,
                [suppression_reason or "suppressed_active_contact"],
                contact.callsign if contact else None,
            )
            return None
        if (
            sensitivity_support
            or (
                strong_route
                and self.config.detection.immediate_notification_on_clear_ground_match
                and arrival.status == MatchStatus.CONFIRMED
                and arrival.registration is not None
            )
        ):
            confirmed = await self._confirm_pending(contact_key, pending.event_id)
            if confirmed is not None:
                self._record_decision(
                    event,
                    DetectionDecision.CONFIRMED,
                    [
                        *(
                            [
                                "alert sensitivity policy accepted strong non-pilot "
                                "arrival evidence"
                            ]
                            if sensitivity_support
                            else ["pilot arrival phraseology and clear KMRY ADS-B match"]
                        ),
                        "immediate high-recall notification enabled",
                    ],
                    contact.callsign if contact else None,
                )
                return confirmed
        self._record_decision(
            event,
            DetectionDecision.PENDING,
            [
                "Waiting "
                f"{self.config.detection.destination_confirmation_delay_seconds:g} seconds "
                "for a possible correction",
                *(
                    [
                        "controller route candidate requires linked pilot evidence "
                        "before notification"
                    ]
                    if controller_only_route
                    else []
                ),
            ],
            contact.callsign if contact else None,
        )
        return pending
