from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

MODEL_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?$"


def validate_model_identifier(value: str | None) -> str | None:
    import re

    if value is None:
        return None
    value = value.strip()
    if not re.fullmatch(MODEL_ID_PATTERN, value) or ".." in value:
        raise ValueError("model must be a faster-whisper model name or owner/repository identifier")
    return value


class AirportConfig(BaseModel):
    icao: str = "KMRY"
    latitude: float = 36.587
    longitude: float = -121.843
    adsb_radius_nm: float = 5


class DestinationConfig(BaseModel):
    canonical_name: str = "Monterey Jet Center"
    phrases: list[str] = ["monterey jet center", "monterey jet", "the jet center", "jet center"]
    taxi_context_phrases: list[str] = [
        "taxi",
        "parking",
        "park",
        "going to",
        "headed to",
        "destination",
        "say parking",
        "where are you parking",
    ]


class KnownDestinationConfig(BaseModel):
    canonical_name: str
    aliases: list[str]


class AdsbPromptConfig(BaseModel):
    enabled: bool = True
    refresh_seconds: float = Field(default=7, ge=1)
    max_aircraft: int = Field(default=5, ge=1, le=20)


class SpeechConfig(BaseModel):
    provider: str = "faster_whisper"
    model: str = "small.en"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"
    beam_size: int = Field(default=5, ge=1)
    temperature: float = Field(default=0.0, ge=0)
    condition_on_previous_text: bool = False
    vad_filter: bool = False
    use_internal_vad: bool | None = None
    fallback_model: str | None = None
    allow_model_fallback: bool = False
    use_static_aviation_prompt: bool = True
    use_adsb_dynamic_prompt: bool = True
    adsb_prompt: AdsbPromptConfig = AdsbPromptConfig()
    initial_prompt: str = (
        "Monterey Ground communications at KMRY. Aircraft callsigns and registrations use "
        "aviation phraseology. Monterey Jet Center. Del Monte Aviation. Runway Two Eight Left. "
        "Runway One Zero Right. Taxiways Alpha, Bravo, Charlie, Delta, Echo, Foxtrot, Golf, "
        "Hotel, Juliet, Kilo, Lima, Mike. Say parking. Taxi via Alpha and Echo. Hold short. "
        "Continue down the runway. Turn left at Foxtrot. November, Alpha, Bravo, Charlie, "
        "Delta, Echo, Foxtrot, Golf, Hotel, India, Juliet, Kilo, Lima, Mike, Oscar, Papa, "
        "Quebec, Romeo, Sierra, Tango, Uniform, Victor, Whiskey, X-ray, Yankee, Zulu. Zero, "
        "one, two, three, four, five, six, seven, eight, niner."
    )

    _validate_model = field_validator("model", "fallback_model")(validate_model_identifier)

    @model_validator(mode="after")
    def _apply_internal_vad_alias(self) -> SpeechConfig:
        if self.use_internal_vad is not None:
            self.vad_filter = self.use_internal_vad
        return self


class SpeechModelOverride(BaseModel):
    device: str | None = None
    compute_type: str | None = None
    beam_size: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    condition_on_previous_text: bool | None = None
    vad_filter: bool | None = None
    use_internal_vad: bool | None = None


class SpeechPerformanceConfig(BaseModel):
    excellent_rtf_max: float = Field(default=0.50, gt=0)
    acceptable_rtf_max: float = Field(default=0.80, gt=0)
    risky_rtf_max: float = Field(default=1.00, gt=0)
    live_warning_rtf: float = Field(default=0.80, gt=0)
    rolling_window: int = Field(default=5, ge=1, le=100)

    @field_validator("acceptable_rtf_max")
    @classmethod
    def _acceptable_above_excellent(cls, value: float, info: Any) -> float:
        if value <= info.data.get("excellent_rtf_max", 0):
            raise ValueError("acceptable_rtf_max must exceed excellent_rtf_max")
        return value

    @field_validator("risky_rtf_max")
    @classmethod
    def _risky_above_acceptable(cls, value: float, info: Any) -> float:
        if value <= info.data.get("acceptable_rtf_max", 0):
            raise ValueError("risky_rtf_max must exceed acceptable_rtf_max")
        return value


class BenchmarkConfig(BaseModel):
    model_timeout_seconds: float = Field(default=180, gt=0)


class AudioPreprocessingConfig(BaseModel):
    enabled: bool = True
    sample_rate: int = 16000
    high_pass_hz: float = Field(default=275, ge=0)
    low_pass_hz: float = Field(default=3800, gt=0)
    normalize: bool = True
    target_peak_dbfs: float = Field(default=-3.0, le=0)
    noise_reduction: bool = False
    limiter: bool = True


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    frame_duration_ms: int = 30
    vad_aggressiveness: int = 2
    pre_roll_ms: int = 250
    end_silence_ms: int = 350
    max_transmission_seconds: float = 8
    save_debug_audio: bool = False
    debug_audio_directory: Path = Path("data/debug_audio")
    preprocessing: AudioPreprocessingConfig = AudioPreprocessingConfig()


class AudioClassifierConfig(BaseModel):
    enabled: bool = False
    backend: Literal["local", "rule_based", "mock"] = "local"
    model_path: Path = Path("data/models/atc_intent_classifier")
    sample_rate: int = Field(default=16000, gt=0)
    confidence_threshold: float = Field(default=0.80, ge=0, le=1)
    strong_confidence_threshold: float = Field(default=0.92, ge=0, le=1)
    correction_threshold: float = Field(default=0.85, ge=0, le=1)
    noise_rejection_threshold: float = Field(default=0.65, ge=0, le=1)
    inference_timeout_seconds: float = Field(default=5, gt=0)
    allow_whisper_fallback: bool = True
    require_adsb_for_notification: bool = True
    log_raw_scores: bool = False


class AudioClassifierLabelsConfig(BaseModel):
    destinations: list[str] = [
        "monterey_jet_center",
        "del_monte_aviation",
        "other_or_unknown_destination",
        "no_destination",
    ]
    intents: list[str] = [
        "taxi_or_route_to_destination",
        "parking_statement",
        "parking_prompt_response",
        "correction_or_destination_change",
        "weak_destination_mention",
        "no_relevant_intent",
        "unintelligible_or_noise",
    ]


class TrainingDataConfig(BaseModel):
    enabled: bool = False
    directory: Path = Path("data/training_clips")
    save_uncertain_only: bool = True
    save_all_candidate_events: bool = False
    save_audio: bool = True
    save_transcript: bool = True
    save_adsb_context: bool = True
    save_classifier_scores: bool = True
    redact_pairing_tokens: bool = True


class AudioAugmentationConfig(BaseModel):
    enabled: bool = False
    random_seed: int = 457
    noise_probability: float = Field(default=0.35, ge=0, le=1)
    maximum_noise_amplitude: float = Field(default=0.025, ge=0, le=0.25)
    gain_range: tuple[float, float] = (0.8, 1.2)
    clipping_probability: float = Field(default=0.1, ge=0, le=1)
    dropout_probability: float = Field(default=0.1, ge=0, le=1)
    band_pass_probability: float = Field(default=0.25, ge=0, le=1)
    band_pass_low_hz: float = Field(default=250, ge=0)
    band_pass_high_hz: float = Field(default=4000, gt=0)
    compression_probability: float = Field(default=0.2, ge=0, le=1)
    frequency_response_probability: float = Field(default=0.2, ge=0, le=1)
    speed_factors: list[float] = [0.97, 1.0, 1.03]


class DecisionFusionConfig(BaseModel):
    classifier_primary: bool = True
    allow_whisper_only_alerts: bool = False
    preserve_strong_pending_event_on_low_quality_followup: bool = True
    require_adsb_resolution: bool = True


class TrafficFilterConfig(BaseModel):
    enabled: bool = True
    ignore_scheduled_airlines: bool = True
    ignored_operator_names: list[str] = Field(default_factory=list)
    ignored_icao_designators: list[str] = Field(default_factory=list)
    ignored_iata_codes: list[str] = Field(default_factory=list)
    ignored_callsign_prefixes: list[str] = Field(default_factory=list)
    allowed_operator_overrides: list[str] = ["JSX"]
    allowed_icao_designators: list[str] = ["JSX"]
    allowed_iata_codes: list[str] = Field(default_factory=list)
    log_filtered_aircraft: bool = True


class IntentDetectionConfig(BaseModel):
    destination_phrase_threshold: float = Field(default=0.82, ge=0, le=1)
    route_context_threshold: float = Field(default=0.72, ge=0, le=1)
    weak_phrase_with_strong_route_threshold: float = Field(default=0.68, ge=0, le=1)
    allow_partial_jet_center_match: bool = True
    allow_contextual_destination_inference: bool = True
    preserve_pending_on_low_quality_followup: bool = True


class AdsbConfig(BaseModel):
    provider: Literal["adsb_lol", "adsb_fi"] = "adsb_lol"
    base_url: str = "https://api.adsb.lol"
    polling_interval_seconds: float = 10
    timeout_seconds: float = 5
    maximum_seen_seconds: float = 20
    maximum_taxi_speed_knots: float = 40


class DetectionConfig(BaseModel):
    alert_sensitivity: Literal["conservative", "balanced", "never_miss"] = "conservative"
    context_window_seconds: float = 120
    alert_threshold: float = 0.80
    ambiguity_margin: float = 0.15
    duplicate_suppression_seconds: float = 300
    notify_on_unresolved: bool = False
    destination_confirmation_delay_seconds: float = Field(default=8, ge=0)
    destination_correction_window_seconds: float = Field(default=20, ge=0)
    strong_destination_threshold: float = Field(default=0.90, ge=0, le=1)
    route_intent_threshold: float = Field(default=0.80, ge=0, le=1)
    fuzzy_intent_matching: bool = True
    fuzzy_intent_minimum_score: int = Field(default=85, ge=0, le=100)
    pending_followup_link_seconds: float = Field(default=20, gt=0)
    alert_on_any_destination_mention: bool = False
    immediate_notification_on_clear_ground_match: bool = False


class AdsbMatchingConfig(BaseModel):
    radius_nm: float = Field(default=5, gt=0)
    maximum_seen_seconds: float = Field(default=20, ge=0)
    maximum_ground_speed_knots: float = Field(default=40, ge=0)
    maximum_low_altitude_feet: float = Field(default=500, ge=0)
    require_recent_position: bool = True
    allow_unique_ground_candidate_fallback: bool = True
    unique_candidate_minimum_confidence: float = Field(default=0.75, ge=0, le=1)
    ambiguity_margin: float = Field(default=0.15, ge=0, le=1)
    fuzzy_callsign_matching: bool = True
    fuzzy_minimum_score: int = Field(default=85, ge=0, le=100)
    taxi_speed_min_knots: float = Field(default=1, ge=0)
    taxi_speed_max_knots: float = Field(default=35, ge=0)
    stopped_speed_max_knots: float = Field(default=1, ge=0)
    recently_landed_seconds: float = Field(default=180, gt=0)
    airborne_altitude_threshold_ft: float = Field(default=500, ge=0)
    minimum_track_points: int = Field(default=2, ge=1)


class AdsbTrackingConfig(BaseModel):
    enabled: bool = True
    history_seconds: float = Field(default=180, gt=0)
    stale_after_seconds: float = Field(default=15, gt=0)
    purge_after_seconds: float = Field(default=300, gt=0)
    minimum_history_points: int = Field(default=2, ge=1)
    debug_snapshot_enabled: bool = False
    debug_snapshot_path: Path = Path("data/adsb_tracker_snapshot.json")


class CircleGeofenceConfig(BaseModel):
    type: Literal["circle"] = "circle"
    latitude: float | None = None
    longitude: float | None = None
    radius_nm: float = Field(default=2, gt=0)


class PolygonGeofenceConfig(BaseModel):
    type: Literal["polygon"] = "polygon"
    coordinates: list[tuple[float, float]] = Field(min_length=3)


GeofenceConfig = CircleGeofenceConfig | PolygonGeofenceConfig


class AdsbGeofencesConfig(BaseModel):
    kmry_airport: CircleGeofenceConfig = CircleGeofenceConfig(radius_nm=2)
    runway_area: GeofenceConfig | None = None
    movement_area: GeofenceConfig | None = None
    monterey_jet_center: GeofenceConfig | None = None
    fbo_geofences: dict[str, GeofenceConfig] = Field(default_factory=dict)
    approach_ring: GeofenceConfig | None = None


class AdsbScoringConfig(BaseModel):
    inside_airport: float = 25
    on_ground: float = 25
    recently_landed: float = 30
    taxi_speed: float = 15
    moving_toward_jet_center: float = 25
    near_jet_center: float = 20
    spoken_registration_exact: float = 30
    spoken_registration_fuzzy: float = 15
    spoken_callsign_fuzzy: float = 10
    recent: float = 10
    stale: float = -40
    airborne_climbing: float = -50
    moving_away: float = -25
    departing: float = -50
    incomplete: float = -10
    duplicate_registration: float = -30


class AdsbDecisionConfig(BaseModel):
    minimum_score: float = 50
    minimum_margin: float = Field(default=15, ge=0)
    maximum_candidate_age_seconds: float = Field(default=15, gt=0)
    require_on_ground_or_recently_landed: bool = True
    correlation_window_seconds: float | None = Field(default=None, ge=0)


class NotificationsConfig(BaseModel):
    send_uncertain_alerts: bool = False
    send_unidentified_arrival_alerts: bool = True
    allow_repeated_spoken_registration_fallback: bool = True
    minimum_spoken_registration_observations: int = Field(default=2, ge=2)


class EventHistoryConfig(BaseModel):
    maximum_events: int = Field(default=1000, ge=100, le=10000)


class NtfyConfig(BaseModel):
    enabled: bool = False
    server_url: str = "https://ntfy.sh"
    topic: str = ""
    authorization: str = ""

    @field_validator("server_url")
    @classmethod
    def _validate_server_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("https://", "http://")):
            raise ValueError("ntfy.server_url must use http:// or https://")
        return value

    @field_validator("topic")
    @classmethod
    def _validate_topic(cls, value: str) -> str:
        import re

        value = value.strip()
        if value and not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("ntfy.topic may contain only letters, numbers, _ and -")
        return value

    @model_validator(mode="after")
    def _require_topic_when_enabled(self) -> NtfyConfig:
        if self.enabled and not self.topic:
            raise ValueError("ntfy.topic is required when ntfy is enabled")
        return self


class LoggingConfig(BaseModel):
    live_transcripts: bool = True
    detection_decisions: bool = True
    notification_delivery: bool = True
    verbose_transcripts: bool = False
    whisper_quality: bool = True
    adsb_candidates: bool = True
    format: Literal["detailed", "compact"] = "detailed"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    pairing_token_file: Path = Path("data/pairing_token.txt")


class LiveAtcConfig(BaseModel):
    enabled: bool = True
    authorized_player_url: str = "https://www.liveatc.net/hlisten.php?mount=kmry&icao=kmry"
    authorized_mount: str = "kmry"
    authorized_icao: str = "kmry"
    source_label: str = "liveatc_kmry_web_audio"
    audio_websocket_path: str = "/ws/audio"


class AppConfig(BaseModel):
    airport: AirportConfig = AirportConfig()
    destination: DestinationConfig = DestinationConfig()
    destinations: list[KnownDestinationConfig] = Field(
        default_factory=lambda: [
            KnownDestinationConfig(
                canonical_name="Monterey Jet Center",
                aliases=["monterey jet center", "monterey jet", "the jet center", "jet center"],
            ),
            KnownDestinationConfig(
                canonical_name="Del Monte Aviation",
                aliases=["del monte aviation", "del monte"],
            ),
        ]
    )
    speech: SpeechConfig = SpeechConfig()
    speech_model_overrides: dict[str, SpeechModelOverride] = Field(default_factory=dict)
    speech_performance: SpeechPerformanceConfig = SpeechPerformanceConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    audio: AudioConfig = AudioConfig()
    audio_classifier: AudioClassifierConfig = AudioClassifierConfig()
    audio_classifier_labels: AudioClassifierLabelsConfig = AudioClassifierLabelsConfig()
    training_data: TrainingDataConfig = TrainingDataConfig()
    audio_augmentation: AudioAugmentationConfig = AudioAugmentationConfig()
    decision_fusion: DecisionFusionConfig = DecisionFusionConfig()
    traffic_filter: TrafficFilterConfig = TrafficFilterConfig()
    intent_detection: IntentDetectionConfig = IntentDetectionConfig()
    adsb: AdsbConfig = AdsbConfig()
    adsb_matching: AdsbMatchingConfig = AdsbMatchingConfig()
    adsb_tracking: AdsbTrackingConfig = AdsbTrackingConfig()
    adsb_geofences: AdsbGeofencesConfig = AdsbGeofencesConfig()
    adsb_scoring: AdsbScoringConfig = AdsbScoringConfig()
    adsb_decision: AdsbDecisionConfig = AdsbDecisionConfig()
    notifications: NotificationsConfig = NotificationsConfig()
    event_history: EventHistoryConfig = EventHistoryConfig()
    ntfy: NtfyConfig = NtfyConfig()
    detection: DetectionConfig = DetectionConfig()
    logging: LoggingConfig = LoggingConfig()
    server: ServerConfig = ServerConfig()
    liveatc: LiveAtcConfig = LiveAtcConfig()

    @field_validator("speech_model_overrides")
    @classmethod
    def _validate_override_model_names(
        cls, value: dict[str, SpeechModelOverride]
    ) -> dict[str, SpeechModelOverride]:
        for model in value:
            validate_model_identifier(model)
        return value

    def speech_for_model(self, model: str) -> SpeechConfig:
        validated = validate_model_identifier(model)
        assert validated is not None
        model = validated
        override = self.speech_model_overrides.get(model)
        updates: dict[str, object] = {"model": model}
        if override:
            updates.update(override.model_dump(exclude_none=True))
            if override.use_internal_vad is not None:
                updates["vad_filter"] = override.use_internal_vad
        return self.speech.model_copy(update=updates)

    @classmethod
    def load(cls, path: Path | None = None) -> AppConfig:
        if path is None:
            return cls()
        with path.open("r", encoding="utf-8") as handle:
            data: Any = yaml.safe_load(handle) or {}
        return cls.model_validate(data)
