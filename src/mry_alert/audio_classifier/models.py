from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DestinationLabel(StrEnum):
    MONTEREY_JET_CENTER = "monterey_jet_center"
    DEL_MONTE_AVIATION = "del_monte_aviation"
    OTHER_OR_UNKNOWN = "other_or_unknown_destination"
    NONE = "no_destination"


class IntentLabel(StrEnum):
    TAXI_OR_ROUTE = "taxi_or_route_to_destination"
    PARKING_STATEMENT = "parking_statement"
    PARKING_PROMPT_RESPONSE = "parking_prompt_response"
    CORRECTION = "correction_or_destination_change"
    WEAK_DESTINATION_MENTION = "weak_destination_mention"
    NONE = "no_relevant_intent"
    NOISE = "unintelligible_or_noise"


class ClassificationContext(BaseModel):
    timestamp: datetime
    source: str
    prior_destination_prompt: bool = False
    nearby_registrations: list[str] = Field(default_factory=list)
    whisper_transcript: str | None = None


class AudioIntentResult(BaseModel):
    destination: DestinationLabel = DestinationLabel.NONE
    destination_confidence: float = Field(default=0.0, ge=0, le=1)
    intent: IntentLabel = IntentLabel.NONE
    intent_confidence: float = Field(default=0.0, ge=0, le=1)
    correction: bool = False
    correction_confidence: float = Field(default=0.0, ge=0, le=1)
    noise_confidence: float = Field(default=0.0, ge=0, le=1)
    callsign_clue: str | None = None
    raw_scores: dict[str, float] = Field(default_factory=dict)
    model_version: str = "disabled"
    reasons: list[str] = Field(default_factory=list)
    inference_seconds: float = Field(default=0.0, ge=0)
