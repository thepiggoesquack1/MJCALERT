from __future__ import annotations

import hashlib
import os
import re
import shutil
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from mry_alert.audio_classifier.models import AudioIntentResult
from mry_alert.config import TrainingDataConfig
from mry_alert.models import TranscriptEvent

TOKEN_PATTERN = re.compile(r"(?i)(token=)[^&\s\"']+")
DETECTION_INTENT_LABELS = {
    "explicit_taxi_request": "taxi_or_route_to_destination",
    "explicit_parking_statement": "parking_statement",
    "parking_prompt_response": "parking_prompt_response",
    "ground_route_to_destination": "taxi_or_route_to_destination",
    "weak_destination_mention": "weak_destination_mention",
    "none": "no_relevant_intent",
}


class ReviewedLabels(BaseModel):
    destination: str
    intent: str
    correction: bool = False
    callsign_or_registration: str | None = None
    unintelligible: bool = False


class TrainingClipMetadata(BaseModel):
    schema_version: int = 2
    clip_id: str
    timestamp: datetime
    sample_rate: int
    wav_file: str | None = None
    original_transcript: str = ""
    normalized_transcript: str = ""
    classifier_output: AudioIntentResult
    adsb_candidates: list[dict[str, Any]] = Field(default_factory=list)
    selected_aircraft: str | None = None
    destination_result: str | None = None
    intent_result: str | None = None
    user_correction_status: str = "pending"
    label_source: str = "model_prediction"
    reviewed: bool = False
    reviewed_labels: ReviewedLabels | None = None
    classifier_version: str = "unknown"
    adsb_supported: bool = False
    hard_negative: bool = False
    classifier_status: Literal["available", "disabled", "unavailable", "failed"] = "available"
    detection_event_id: str | None = None
    detection_decision: str | None = None
    detection_reasons: list[str] = Field(default_factory=list)
    speaker_role: str = "unknown"
    speaker_role_confidence: float = Field(default=0.0, ge=0, le=1)
    destination_candidate: str | None = None
    intent_category: str | None = None
    detected_callsign: str | None = None
    identified_registration: str | None = None
    audio_sha256: str | None = None
    audio_duration_seconds: float | None = Field(default=None, ge=0)
    collection_complete: bool = True


class ClipValidationIssue(BaseModel):
    code: str
    path: Path
    detail: str


class DatasetScan(BaseModel):
    valid_metadata: list[Path] = Field(default_factory=list)
    issues: list[ClipValidationIssue] = Field(default_factory=list)


class TrainingDataCollector:
    def __init__(self, config: TrainingDataConfig) -> None:
        self.config = config

    def _directory(self, name: str) -> Path:
        return self.config.directory / name

    def ensure_layout(self) -> None:
        if not self.config.enabled:
            return
        for name in ("pending", "reviewed", "rejected", "hard_negatives"):
            self._directory(name).mkdir(parents=True, exist_ok=True)

    def pending_count(self) -> int:
        pending = self._directory("pending")
        return len(scan_clips(pending).valid_metadata) if pending.exists() else 0

    def save(
        self,
        pcm: bytes,
        sample_rate: int,
        *,
        transcript: str,
        normalized_transcript: str,
        classification: AudioIntentResult | None,
        classifier_status: Literal["available", "disabled", "unavailable", "failed"] = "available",
        detection_event: TranscriptEvent | None = None,
        adsb_candidates: list[dict[str, Any]] | None = None,
        selected_aircraft: str | None = None,
        adsb_supported: bool = False,
        uncertain: bool = True,
    ) -> Path | None:
        if not self.config.enabled:
            return None
        if self.config.save_uncertain_only and not uncertain:
            return None
        if not self.config.save_all_candidate_events and not uncertain:
            return None
        self.ensure_layout()
        now = datetime.now(UTC)
        clip_id = f"{now:%Y%m%d_%H%M%S_%f}_{uuid4().hex[:12]}"
        pending = self._directory("pending")
        wav_name = f"{clip_id}_clip.wav" if self.config.save_audio else None
        audio_sha256 = hashlib.sha256(pcm).hexdigest() if wav_name else None
        duration_seconds = len(pcm) / 2 / sample_rate if sample_rate > 0 else 0.0
        if wav_name:
            temporary_wav = pending / f".{wav_name}.incomplete"
            final_wav = pending / wav_name
            with wave.open(str(temporary_wav), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(sample_rate)
                output.writeframes(pcm)
            os.replace(temporary_wav, final_wav)
        safe_transcript = (
            TOKEN_PATTERN.sub(r"\1<redacted>", transcript)
            if self.config.redact_pairing_tokens
            else transcript
        )
        classifier_value = classification or AudioIntentResult(
            model_version="unavailable",
            reasons=["audio classifier was not available during collection"],
        )
        metadata = TrainingClipMetadata(
            clip_id=clip_id,
            timestamp=now,
            sample_rate=sample_rate,
            wav_file=wav_name,
            original_transcript=safe_transcript if self.config.save_transcript else "",
            normalized_transcript=(
                TOKEN_PATTERN.sub(r"\1<redacted>", normalized_transcript)
                if self.config.save_transcript
                else ""
            ),
            classifier_output=classifier_value.model_copy(
                update={"raw_scores": classifier_value.raw_scores}
                if self.config.save_classifier_scores
                else {"raw_scores": {}}
            ),
            adsb_candidates=(adsb_candidates or []) if self.config.save_adsb_context else [],
            selected_aircraft=selected_aircraft,
            destination_result=(
                detection_event.destination_candidate
                if detection_event and detection_event.destination_candidate
                else classifier_value.destination.value
            ),
            intent_result=(
                DETECTION_INTENT_LABELS[detection_event.intent_category.value]
                if detection_event
                else classifier_value.intent.value
            ),
            classifier_version=classifier_value.model_version,
            adsb_supported=adsb_supported,
            classifier_status=classifier_status,
            detection_event_id=detection_event.event_id if detection_event else None,
            detection_decision=(
                detection_event.detection_decision.value if detection_event else None
            ),
            detection_reasons=(detection_event.detection_reasons if detection_event else []),
            speaker_role=(detection_event.speaker_role.value if detection_event else "unknown"),
            speaker_role_confidence=(
                detection_event.speaker_role_confidence if detection_event else 0.0
            ),
            destination_candidate=(
                detection_event.destination_candidate if detection_event else None
            ),
            intent_category=(
                DETECTION_INTENT_LABELS[detection_event.intent_category.value]
                if detection_event
                else None
            ),
            detected_callsign=(detection_event.detected_callsign if detection_event else None),
            identified_registration=(
                detection_event.identified_registration if detection_event else None
            ),
            audio_sha256=audio_sha256,
            audio_duration_seconds=duration_seconds,
        )
        path = pending / f"{clip_id}_metadata.json"
        temporary_metadata = pending / f".{clip_id}_metadata.json.incomplete"
        temporary_metadata.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary_metadata, path)
        return path


def load_metadata(path: Path) -> TrainingClipMetadata:
    return TrainingClipMetadata.model_validate_json(path.read_text(encoding="utf-8"))


def review_clip(
    metadata_path: Path,
    labels: ReviewedLabels,
    *,
    reject: bool = False,
    hard_negative: bool = False,
) -> Path:
    metadata = load_metadata(metadata_path)
    if hard_negative:
        labels = labels.model_copy(
            update={
                "destination": "no_destination",
                "intent": (
                    "unintelligible_or_noise" if labels.unintelligible else "no_relevant_intent"
                ),
            }
        )
    target_name = "rejected" if reject else "hard_negatives" if hard_negative else "reviewed"
    target = metadata_path.parent.parent / target_name
    target.mkdir(parents=True, exist_ok=True)
    metadata.reviewed = not reject
    metadata.reviewed_labels = labels
    metadata.label_source = "human_review"
    metadata.user_correction_status = "rejected" if reject else "corrected"
    metadata.hard_negative = hard_negative
    destination = target / metadata_path.name
    if destination.exists():
        raise FileExistsError(f"Reviewed clip already exists: {destination.name}")
    temporary = target / f".{metadata_path.name}.incomplete"
    temporary.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    if metadata.wav_file:
        source_wav = metadata_path.parent / metadata.wav_file
        if source_wav.exists():
            target_wav = target / metadata.wav_file
            if target_wav.exists():
                raise FileExistsError(f"Reviewed audio already exists: {target_wav.name}")
            shutil.move(str(source_wav), str(target_wav))
    metadata_path.unlink()
    return destination


def scan_clips(directory: Path, *, require_reviewed: bool = False) -> DatasetScan:
    """Validate a clip directory without mutating or deleting any artifacts."""
    result = DatasetScan()
    if not directory.exists():
        return result
    metadata_paths = sorted(directory.glob("*_metadata.json"))
    referenced_audio: set[Path] = set()
    seen_hashes: dict[str, Path] = {}
    for path in metadata_paths:
        try:
            metadata = load_metadata(path)
        except (OSError, ValueError) as exc:
            result.issues.append(
                ClipValidationIssue(code="corrupted_metadata", path=path, detail=str(exc))
            )
            continue
        issues: list[ClipValidationIssue] = []
        if not metadata.collection_complete:
            issues.append(
                ClipValidationIssue(
                    code="incomplete_write",
                    path=path,
                    detail="metadata is not marked complete",
                )
            )
        if require_reviewed and (
            not metadata.reviewed
            or metadata.label_source != "human_review"
            or metadata.reviewed_labels is None
        ):
            issues.append(
                ClipValidationIssue(
                    code="not_human_reviewed",
                    path=path,
                    detail="clip has no completed human review",
                )
            )
        if metadata.wav_file:
            wav_path = directory / metadata.wav_file
            referenced_audio.add(wav_path)
            if not wav_path.is_file():
                issues.append(
                    ClipValidationIssue(
                        code="missing_audio",
                        path=path,
                        detail=f"referenced WAV does not exist: {metadata.wav_file}",
                    )
                )
            else:
                try:
                    with wave.open(str(wav_path), "rb") as audio:
                        frames = audio.getnframes()
                        rate = audio.getframerate()
                        payload = audio.readframes(frames)
                    if frames <= 0 or rate <= 0:
                        issues.append(
                            ClipValidationIssue(
                                code="zero_duration_audio",
                                path=wav_path,
                                detail="WAV contains no playable frames",
                            )
                        )
                    digest = hashlib.sha256(payload).hexdigest()
                    if metadata.audio_sha256 and digest != metadata.audio_sha256:
                        issues.append(
                            ClipValidationIssue(
                                code="audio_hash_mismatch",
                                path=wav_path,
                                detail="WAV content does not match its metadata",
                            )
                        )
                    previous = seen_hashes.get(digest)
                    if previous:
                        issues.append(
                            ClipValidationIssue(
                                code="duplicate_clip",
                                path=path,
                                detail=f"audio duplicates {previous.name}",
                            )
                        )
                    else:
                        seen_hashes[digest] = path
                except (OSError, EOFError, wave.Error) as exc:
                    issues.append(
                        ClipValidationIssue(code="corrupted_audio", path=wav_path, detail=str(exc))
                    )
        elif require_reviewed:
            issues.append(
                ClipValidationIssue(
                    code="missing_audio",
                    path=path,
                    detail="reviewed training example has no WAV file",
                )
            )
        if issues:
            result.issues.extend(issues)
        else:
            result.valid_metadata.append(path)

    incomplete_paths = set(directory.glob("*.incomplete")) | set(directory.glob(".*.incomplete"))
    for incomplete in sorted(incomplete_paths):
        result.issues.append(
            ClipValidationIssue(
                code="incomplete_write",
                path=incomplete,
                detail="temporary collection artifact was not committed",
            )
        )
    for wav_path in sorted(directory.glob("*.wav")):
        if wav_path not in referenced_audio:
            result.issues.append(
                ClipValidationIssue(
                    code="orphaned_audio",
                    path=wav_path,
                    detail="WAV has no matching metadata record",
                )
            )
    return result
