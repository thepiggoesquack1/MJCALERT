from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from mry_alert.audio_classifier.models import AudioIntentResult


class MatchStatus(StrEnum):
    CONFIRMED = "confirmed"
    LIKELY = "likely"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


class AlertEventType(StrEnum):
    ARRIVAL = "arrival"
    DESTINATION_CORRECTION = "destination_correction"
    DESTINATION_CANCELLED = "destination_cancelled"


class ConfirmationStatus(StrEnum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    UNRESOLVED = "unresolved"


class OperatorAcknowledgement(StrEnum):
    UNREVIEWED = "unreviewed"
    SEEN = "seen"
    AIRCRAFT_ARRIVED = "aircraft_arrived"
    FALSE_DETECTION = "false_detection"


class DestinationState(StrEnum):
    UNKNOWN = "destination_unknown"
    PENDING = "destination_pending"
    CONFIRMED = "destination_confirmed"
    CORRECTED = "destination_corrected"
    CANCELLED = "destination_cancelled"


class SpeakerRole(StrEnum):
    CONTROLLER = "controller"
    PILOT = "pilot"
    UNKNOWN = "unknown"


class DetectionDecision(StrEnum):
    TRANSCRIBED = "TRANSCRIBED"
    IGNORED = "IGNORED"
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"
    CANCELLED = "CANCELLED"
    CORRECTED = "CORRECTED"


class DestinationIntentCategory(StrEnum):
    EXPLICIT_TAXI_REQUEST = "explicit_taxi_request"
    EXPLICIT_PARKING_STATEMENT = "explicit_parking_statement"
    PARKING_PROMPT_RESPONSE = "parking_prompt_response"
    GROUND_ROUTE_TO_DESTINATION = "ground_route_to_destination"
    WEAK_DESTINATION_MENTION = "weak_destination_mention"
    NONE = "none"


class IdentificationSource(StrEnum):
    SPOKEN_FULL_REGISTRATION = "spoken_full_registration"
    SPOKEN_CALLSIGN_ADSB_MATCH = "spoken_callsign_adsb_match"
    UNIQUE_SUFFIX_ADSB_MATCH = "unique_suffix_adsb_match"
    FUZZY_ADSB_RECOVERY = "fuzzy_adsb_recovery"
    UNIQUE_GROUND_CANDIDATE = "unique_ground_candidate"
    UNRESOLVED = "unresolved"
    ADSB_CORRELATION = "adsb_correlation"


class AdsbDecisionState(StrEnum):
    CONFIRMED = "confirmed"
    LIKELY = "likely"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT_DATA = "insufficient_data"
    NO_CANDIDATE = "no_candidate"
    REJECTED = "rejected"


class TranscriptEvent(BaseModel):
    event_id: str
    timestamp: datetime
    text: str
    normalized_text: str = ""
    source: str = "simulation"
    duration_seconds: float = 0.0
    transcription_confidence: float | None = None
    speaker_role: SpeakerRole = SpeakerRole.UNKNOWN
    speaker_role_confidence: float = Field(default=0.0, ge=0, le=1)
    speaker_role_reasons: list[str] = Field(default_factory=list)
    detected_callsign: str | None = None
    destination_candidate: str | None = None
    destination_candidate_confidence: float | None = Field(default=None, ge=0, le=1)
    detection_decision: DetectionDecision = DetectionDecision.TRANSCRIBED
    detection_reasons: list[str] = Field(default_factory=list)
    intent_category: DestinationIntentCategory = DestinationIntentCategory.NONE
    route_cues: list[str] = Field(default_factory=list)
    normalization_reasons: list[str] = Field(default_factory=list)
    artifact_trimming_reason: str | None = None
    average_log_probability: float | None = None
    no_speech_probability: float | None = Field(default=None, ge=0, le=1)
    transcription_segment_count: int | None = Field(default=None, ge=0)
    transcript_duration_seconds: float | None = Field(default=None, ge=0)
    whisper_quality: str | None = None
    audio_intent: AudioIntentResult | None = None
    fusion_decision: str | None = None
    identification_source: IdentificationSource = IdentificationSource.UNRESOLVED
    identified_registration: str | None = None
    identified_operator: str | None = None
    aircraft_type_code: str | None = None
    aircraft_type_name: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    aircraft_category: str | None = None
    aircraft_type_source: str = "unknown"
    aircraft_type_confidence: float = Field(default=0.0, ge=0, le=1)
    adsb_movement_state: str | None = None
    adsb_candidate_reasons: list[str] = Field(default_factory=list)
    adsb_decision: AdsbDecisionState | None = None
    adsb_winning_score: float | None = None
    adsb_winning_margin: float | None = None
    traffic_filter_decision: str = "allowed"
    traffic_filter_reasons: list[str] = Field(default_factory=list)


class SpokenCallsign(BaseModel):
    original_text: str
    normalized_form: str
    full_registration: str | None = None
    suffix: str | None = None
    aircraft_type_prefix: str | None = None
    operator_callsign: str | None = None
    parse_confidence: float = Field(ge=0, le=1)
    parse_reasons: list[str] = Field(default_factory=list)


class NearbyAircraft(BaseModel):
    hex: str
    registration: str | None = None
    flight: str | None = None
    operator_name: str | None = None
    icao_designator: str | None = None
    iata_code: str | None = None
    aircraft_type: str | None = None
    aircraft_type_code: str | None = None
    aircraft_type_name: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    aircraft_category: str | None = None
    aircraft_type_source: str = "unknown"
    aircraft_type_confidence: float = Field(default=0.0, ge=0, le=1)
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | str | None = None
    on_ground: bool = False
    ground_speed: float | None = None
    seconds_since_seen: float | None = None
    distance_nm: float | None = None
    track: float | None = None
    vertical_rate: float | None = None
    squawk: str | None = None
    source_timestamp: datetime | None = None


class AircraftMatch(BaseModel):
    aircraft: NearbyAircraft | None = None
    confidence: float = Field(ge=0, le=1)
    status: MatchStatus
    match_reasons: list[str] = Field(default_factory=list)
    alternative_candidates: list[NearbyAircraft] = Field(default_factory=list)
    registration: str | None = None
    raw_callsign_text: str | None = None
    normalized_callsign: str | None = None
    identification_source: IdentificationSource = IdentificationSource.UNRESOLVED
    candidate_scores: list[str] = Field(default_factory=list)
    adsb_snapshot_timestamp: datetime | None = None
    adsb_decision: AdsbDecisionState | None = None
    winning_score: float | None = None
    second_best_score: float | None = None
    winning_margin: float | None = None
    movement_state: str | None = None
    inside_fbo_geofence: bool = False
    fbo_geofence_names: list[str] = Field(default_factory=list)
    was_stationary_at_fbo: bool = False
    moving_away_from_destination: bool = False
    recently_landed: bool = False


class AlertEvent(BaseModel):
    event_id: str
    timestamp: datetime
    destination: str
    registration: str | None = None
    spoken_callsign: str
    aircraft_type: str | None = None
    operator_name: str | None = None
    aircraft_type_code: str | None = None
    aircraft_type_name: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    aircraft_category: str | None = None
    aircraft_type_source: str = "unknown"
    aircraft_type_confidence: float = Field(default=0.0, ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    status: MatchStatus
    transcript_excerpt: str
    match_reasons: list[str] = Field(default_factory=list)
    alternative_registrations: list[str] = Field(default_factory=list)
    event_type: AlertEventType = AlertEventType.ARRIVAL
    previous_destination: str | None = None
    corrected_destination: str | None = None
    original_event_id: str | None = None
    confirmation_status: ConfirmationStatus = ConfirmationStatus.CONFIRMED
    speaker_role: SpeakerRole = SpeakerRole.UNKNOWN
    speaker_role_confidence: float = Field(default=0.0, ge=0, le=1)
    speaker_role_reasons: list[str] = Field(default_factory=list)
    identification_source: IdentificationSource = IdentificationSource.UNRESOLVED
    adsb_winning_score: float | None = None
    adsb_winning_margin: float | None = None
    adsb_movement_state: str | None = None
    intent: DestinationIntentCategory = DestinationIntentCategory.NONE
    classifier_confidence: float | None = Field(default=None, ge=0, le=1)
    whisper_confidence: float | None = Field(default=None, ge=0, le=1)
    decoder_confidence: str | None = None
    test: bool = False


class NotificationRecord(BaseModel):
    event_id: str
    timestamp: datetime
    sent_at: datetime
    destination: str
    registration: str | None = None
    spoken_callsign: str
    aircraft_type: str | None = None
    operator_name: str | None = None
    aircraft_type_code: str | None = None
    aircraft_type_name: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    aircraft_category: str | None = None
    aircraft_type_source: str = "unknown"
    aircraft_type_confidence: float = Field(default=0.0, ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    status: MatchStatus
    confirmation_status: ConfirmationStatus
    transcript_excerpt: str
    match_reasons: list[str] = Field(default_factory=list)
    test: bool = False
    connected_clients: int = Field(ge=0)
    delivered_clients: int = Field(ge=0)
    failed_clients: int = Field(ge=0)


class PendingDestinationEvent(BaseModel):
    event_id: str
    timestamp: datetime
    destination: str
    registration: str | None = None
    spoken_callsign: str
    aircraft_type: str | None = None
    operator_name: str | None = None
    aircraft_type_code: str | None = None
    aircraft_type_name: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    aircraft_category: str | None = None
    aircraft_type_source: str = "unknown"
    aircraft_type_confidence: float = Field(default=0.0, ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    status: MatchStatus
    transcript_excerpt: str
    match_reasons: list[str] = Field(default_factory=list)
    alternative_registrations: list[str] = Field(default_factory=list)
    event_type: AlertEventType = AlertEventType.ARRIVAL
    previous_destination: str | None = None
    corrected_destination: str | None = None
    original_event_id: str | None = None
    confirmation_status: ConfirmationStatus = ConfirmationStatus.PENDING
    speaker_role: SpeakerRole = SpeakerRole.UNKNOWN
    speaker_role_confidence: float = Field(default=0.0, ge=0, le=1)
    speaker_role_reasons: list[str] = Field(default_factory=list)
    identification_source: IdentificationSource = IdentificationSource.UNRESOLVED
    adsb_winning_score: float | None = None
    adsb_winning_margin: float | None = None
    adsb_movement_state: str | None = None
    intent: DestinationIntentCategory = DestinationIntentCategory.NONE
    classifier_confidence: float | None = Field(default=None, ge=0, le=1)
    whisper_confidence: float | None = Field(default=None, ge=0, le=1)
    decoder_confidence: str | None = None
    contact_key: str


class SessionEventRecord(BaseModel):
    event_id: str
    transition_type: str
    timestamp: datetime
    registration: str | None = None
    spoken_callsign: str | None = None
    operator_name: str | None = None
    aircraft_type: str | None = None
    aircraft_type_code: str | None = None
    aircraft_type_name: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    aircraft_category: str | None = None
    aircraft_type_source: str = "unknown"
    aircraft_type_confidence: float = Field(default=0.0, ge=0, le=1)
    destination: str | None = None
    intent: str = "none"
    direction_state: str = "ambiguous"
    adsb_movement_state: str | None = None
    adsb_score: float | None = None
    winning_margin: float | None = None
    final_decision: str
    notification_status: str = "not_sent"
    chrome_delivery_result: str = "not_attempted"
    ntfy_delivery_result: str = "not_attempted"
    decision_reasons: list[str] = Field(default_factory=list)
    transcript_excerpt: str = ""
    classifier_confidence: float | None = Field(default=None, ge=0, le=1)
    whisper_confidence: float | None = Field(default=None, ge=0, le=1)
    decoder_confidence: str | None = None
    original_event_id: str | None = None
    test: bool = False
    operator_acknowledgement: OperatorAcknowledgement = OperatorAcknowledgement.UNREVIEWED
    acknowledged_at: datetime | None = None


class EventAcknowledgementRequest(BaseModel):
    acknowledgement: OperatorAcknowledgement


class SessionHistoryResponse(BaseModel):
    session_id: str
    events: list[SessionEventRecord] = Field(default_factory=list)
