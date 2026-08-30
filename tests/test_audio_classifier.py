from __future__ import annotations

import json
import time
import wave
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

import mry_alert.audio_classifier.dataset as dataset_module
from mry_alert.adsb.mock import MockNearbyAircraftProvider
from mry_alert.audio_classifier.base import (
    DisabledAudioIntentClassifier,
    MockAudioIntentClassifier,
)
from mry_alert.audio_classifier.dataset import (
    ReviewedLabels,
    TrainingDataCollector,
    load_metadata,
    review_clip,
    scan_clips,
)
from mry_alert.audio_classifier.features import (
    LocalTrainableAudioClassifier,
    RuleBasedFeatureClassifier,
)
from mry_alert.audio_classifier.models import (
    AudioIntentResult,
    ClassificationContext,
    DestinationLabel,
    IntentLabel,
)
from mry_alert.audio_classifier.training import (
    augment_pcm,
    binary_metrics,
    confusion_matrix,
    train_classifier,
)
from mry_alert.cli import build_parser
from mry_alert.config import (
    AppConfig,
    AudioAugmentationConfig,
    AudioClassifierConfig,
    DecisionFusionConfig,
    DetectionConfig,
    TrainingDataConfig,
)
from mry_alert.detection.engine import DetectionEngine
from mry_alert.models import NearbyAircraft, PendingDestinationEvent, TranscriptEvent


def context() -> ClassificationContext:
    return ClassificationContext(timestamp=datetime.now(UTC), source="test")


def classification(
    destination: DestinationLabel = DestinationLabel.MONTEREY_JET_CENTER,
    intent: IntentLabel = IntentLabel.TAXI_OR_ROUTE,
    confidence: float = 0.97,
) -> AudioIntentResult:
    return AudioIntentResult(
        destination=destination,
        destination_confidence=confidence,
        intent=intent,
        intent_confidence=confidence,
        correction=intent == IntentLabel.CORRECTION,
        correction_confidence=confidence if intent == IntentLabel.CORRECTION else 0,
        model_version="test-v1",
    )


def event(result: AudioIntentResult, text: str = "NASA Jet Center.") -> TranscriptEvent:
    return TranscriptEvent(
        event_id=str(uuid4()),
        timestamp=datetime.now(UTC),
        text=text,
        source="liveatc_kmry_web_audio",
        audio_intent=result,
    )


def classifier_config() -> AppConfig:
    return AppConfig(
        audio_classifier=AudioClassifierConfig(enabled=True),
        decision_fusion=DecisionFusionConfig(
            classifier_primary=True,
            allow_whisper_only_alerts=False,
            require_adsb_resolution=True,
        ),
        detection=DetectionConfig(destination_confirmation_delay_seconds=0.01),
    )


def nearby() -> NearbyAircraft:
    return NearbyAircraft(
        hex="abc",
        registration="N627S",
        flight="N627S",
        on_ground=True,
        ground_speed=6,
        seconds_since_seen=1,
    )


def test_classifier_interface_and_disabled_fallback() -> None:
    disabled = DisabledAudioIntentClassifier()
    result = disabled.classify(b"", 16000, context())
    assert result.destination == DestinationLabel.NONE
    assert result.model_version == "disabled"
    mock = MockAudioIntentClassifier(classification())
    assert mock.classify(b"pcm", 16000, context()).destination_confidence == 0.97


def test_rule_feature_classifier_rejects_empty_audio_without_inventing_intent() -> None:
    result = RuleBasedFeatureClassifier().classify(b"", 16000, context())
    assert result.intent == IntentLabel.NOISE
    assert result.destination == DestinationLabel.NONE


@pytest.mark.asyncio
async def test_strong_classifier_overrides_bad_whisper_and_requires_adsb() -> None:
    engine = DetectionEngine(classifier_config(), MockNearbyAircraftProvider([nearby()]))
    result = await engine.process(event(classification()))
    assert isinstance(result, PendingDestinationEvent)
    assert result.registration == "N627S"
    assert "audio classifier identified Monterey Jet Center" in result.match_reasons
    await engine.close()


@pytest.mark.asyncio
async def test_del_monte_classifier_suppresses_monterey_alert() -> None:
    engine = DetectionEngine(classifier_config(), MockNearbyAircraftProvider([nearby()]))
    result = await engine.process(event(classification(DestinationLabel.DEL_MONTE_AVIATION)))
    assert result is None
    await engine.close()


@pytest.mark.asyncio
async def test_noise_and_weak_destination_are_suppressed() -> None:
    engine = DetectionEngine(classifier_config(), MockNearbyAircraftProvider([nearby()]))
    noisy = classification(DestinationLabel.NONE, IntentLabel.NOISE)
    noisy.noise_confidence = 0.99
    assert await engine.process(event(noisy)) is None
    assert await engine.process(event(classification(confidence=0.6))) is None
    await engine.close()


@pytest.mark.asyncio
async def test_adsb_required_prevents_unresolved_notification() -> None:
    engine = DetectionEngine(classifier_config(), MockNearbyAircraftProvider([]))
    result = await engine.process(event(classification()))
    assert result is None
    await engine.close()


@pytest.mark.asyncio
async def test_classifier_correction_changes_only_linked_pending_contact() -> None:
    engine = DetectionEngine(classifier_config(), MockNearbyAircraftProvider([nearby()]))
    first = event(
        classification(),
        "November six two seven sierra request taxi.",
    )
    assert isinstance(await engine.process(first), PendingDestinationEvent)
    correction = classification(DestinationLabel.DEL_MONTE_AVIATION, IntentLabel.CORRECTION)
    changed = await engine.process(event(correction, "November six two seven sierra correction."))
    assert isinstance(changed, PendingDestinationEvent)
    assert changed.corrected_destination == "Del Monte Aviation"
    await engine.close()


def test_data_collection_disabled_by_default(tmp_path: Path) -> None:
    collector = TrainingDataCollector(TrainingDataConfig(directory=tmp_path))
    assert (
        collector.save(
            b"\0\0",
            16000,
            transcript="test",
            normalized_transcript="test",
            classification=classification(),
        )
        is None
    )
    assert list(tmp_path.iterdir()) == []


def test_saved_schema_redaction_and_human_review_precedence(tmp_path: Path) -> None:
    collector = TrainingDataCollector(TrainingDataConfig(enabled=True, directory=tmp_path))
    metadata_path = collector.save(
        b"\0\0" * 100,
        16000,
        transcript="ws://localhost/ws?token=secret",
        normalized_transcript="token=secret",
        classification=classification(confidence=0.6),
    )
    assert metadata_path is not None
    metadata = load_metadata(metadata_path)
    assert "secret" not in metadata.original_transcript
    assert not metadata.reviewed
    assert metadata.label_source == "model_prediction"
    reviewed = review_clip(
        metadata_path,
        ReviewedLabels(
            destination=DestinationLabel.DEL_MONTE_AVIATION.value,
            intent=IntentLabel.CORRECTION.value,
            correction=True,
        ),
    )
    final = load_metadata(reviewed)
    assert final.reviewed
    assert final.label_source == "human_review"
    assert final.reviewed_labels is not None
    assert final.reviewed_labels.destination == DestinationLabel.DEL_MONTE_AVIATION


def write_reviewed_example(root: Path, clip_id: str, destination: str, intent: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    wav_name = f"{clip_id}_clip.wav"
    samples = (np.sin(np.linspace(0, 20, 1600)) * 1000).astype("<i2")
    with wave.open(str(root / wav_name), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(samples.tobytes())
    payload = {
        "schema_version": 1,
        "clip_id": clip_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "sample_rate": 16000,
        "wav_file": wav_name,
        "classifier_output": classification().model_dump(mode="json"),
        "reviewed": True,
        "label_source": "human_review",
        "reviewed_labels": {
            "destination": destination,
            "intent": intent,
            "correction": intent == IntentLabel.CORRECTION,
        },
        "classifier_version": "old",
    }
    (root / f"{clip_id}_metadata.json").write_text(json.dumps(payload), encoding="utf-8")


def test_training_uses_reviewed_labels_and_records_class_weights(tmp_path: Path) -> None:
    reviewed = tmp_path / "reviewed"
    write_reviewed_example(
        reviewed,
        "one",
        DestinationLabel.MONTEREY_JET_CENTER,
        IntentLabel.TAXI_OR_ROUTE,
    )
    write_reviewed_example(
        reviewed,
        "two",
        DestinationLabel.NONE,
        IntentLabel.NONE,
    )
    report = train_classifier(reviewed, tmp_path / "model", AudioAugmentationConfig())
    assert report["label_policy"] == "human-reviewed-only"
    assert report["class_weights"]["destination"]
    assert (tmp_path / "model/model.json").is_file()


def test_training_rejects_unreviewed_predictions(tmp_path: Path) -> None:
    collector = TrainingDataCollector(TrainingDataConfig(enabled=True, directory=tmp_path))
    collector.save(
        b"\0\0" * 100,
        16000,
        transcript="",
        normalized_transcript="",
        classification=classification(),
    )
    with pytest.raises(ValueError, match="No human-reviewed"):
        train_classifier(
            tmp_path / "pending",
            tmp_path / "model",
            AudioAugmentationConfig(),
        )


def test_model_loading_failure_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Unable to load"):
        LocalTrainableAudioClassifier(tmp_path / "missing-model")


def test_training_command_argument_validation() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train-audio-classifier"])
    parsed = build_parser().parse_args(
        [
            "train-audio-classifier",
            "--dataset",
            "reviewed",
            "--output",
            "model",
        ]
    )
    assert parsed.command == "train-audio-classifier"


def test_hard_negative_queue_is_separate_and_reviewed(tmp_path: Path) -> None:
    collector = TrainingDataCollector(TrainingDataConfig(enabled=True, directory=tmp_path))
    metadata_path = collector.save(
        b"\0\0" * 100,
        16000,
        transcript="false jet center activation",
        normalized_transcript="false jet center activation",
        classification=classification(confidence=0.55),
    )
    assert metadata_path is not None
    reviewed = review_clip(
        metadata_path,
        ReviewedLabels(
            destination=DestinationLabel.NONE,
            intent=IntentLabel.NONE,
        ),
        hard_negative=True,
    )
    metadata = load_metadata(reviewed)
    assert reviewed.parent.name == "hard_negatives"
    assert metadata.hard_negative
    assert metadata.reviewed
    assert metadata.reviewed_labels is not None
    assert metadata.reviewed_labels.destination == DestinationLabel.NONE
    assert metadata.reviewed_labels.intent == IntentLabel.NONE


def test_approved_hard_negative_is_included_in_training(tmp_path: Path) -> None:
    reviewed = tmp_path / "reviewed"
    write_reviewed_example(
        reviewed,
        "positive",
        DestinationLabel.MONTEREY_JET_CENTER,
        IntentLabel.TAXI_OR_ROUTE,
    )
    collector = TrainingDataCollector(TrainingDataConfig(enabled=True, directory=tmp_path))
    metadata_path = collector.save(
        b"\x01\x00" * 800,
        16000,
        transcript="at the Jet Center request taxi",
        normalized_transcript="at the jet center request taxi",
        classification=None,
        classifier_status="disabled",
    )
    assert metadata_path is not None
    review_clip(
        metadata_path,
        ReviewedLabels(
            destination=DestinationLabel.MONTEREY_JET_CENTER,
            intent=IntentLabel.PARKING_STATEMENT,
        ),
        hard_negative=True,
    )

    report = train_classifier(reviewed, tmp_path / "model", AudioAugmentationConfig())

    assert report["training_examples"] == 2


def test_collection_ids_are_unique_and_raw_audio_respects_config(tmp_path: Path) -> None:
    collector = TrainingDataCollector(
        TrainingDataConfig(enabled=True, directory=tmp_path, save_audio=False)
    )
    paths = [
        collector.save(
            b"\0\0" * 100,
            16000,
            transcript="request taxi",
            normalized_transcript="request taxi",
            classification=None,
            classifier_status="disabled",
        )
        for _ in range(2)
    ]

    assert all(paths)
    assert paths[0] != paths[1]
    assert not list(tmp_path.rglob("*.wav"))


def test_dataset_scan_reports_invalid_and_duplicate_clips(tmp_path: Path) -> None:
    collector = TrainingDataCollector(TrainingDataConfig(enabled=True, directory=tmp_path))
    for transcript in ("first", "duplicate"):
        assert collector.save(
            b"\x01\x00" * 100,
            16000,
            transcript=transcript,
            normalized_transcript=transcript,
            classification=None,
            classifier_status="unavailable",
        )
    pending = tmp_path / "pending"
    (pending / "orphan.wav").write_bytes(b"not a wave")
    (pending / ".interrupted_metadata.json.incomplete").write_text("{", encoding="utf-8")
    (pending / "broken_metadata.json").write_text("{", encoding="utf-8")

    scan = scan_clips(pending)
    codes = {issue.code for issue in scan.issues}

    assert "duplicate_clip" in codes
    assert "orphaned_audio" in codes
    assert "incomplete_write" in codes
    assert "corrupted_metadata" in codes
    assert len(scan.valid_metadata) == 1


def test_dataset_scan_rejects_missing_corrupt_and_zero_duration_audio(tmp_path: Path) -> None:
    pending = tmp_path / "pending"
    pending.mkdir(parents=True)
    for clip_id, wav_name in (
        ("missing", "missing.wav"),
        ("corrupt", "corrupt.wav"),
        ("zero", "zero.wav"),
    ):
        payload = {
            "clip_id": clip_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "sample_rate": 16000,
            "wav_file": wav_name,
            "classifier_output": classification().model_dump(mode="json"),
        }
        (pending / f"{clip_id}_metadata.json").write_text(json.dumps(payload), encoding="utf-8")
    (pending / "corrupt.wav").write_bytes(b"not a wave")
    with wave.open(str(pending / "zero.wav"), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)

    codes = {issue.code for issue in scan_clips(pending).issues}

    assert {"missing_audio", "corrupted_audio", "zero_duration_audio"} <= codes


def test_interrupted_collection_is_not_marked_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collector = TrainingDataCollector(TrainingDataConfig(enabled=True, directory=tmp_path))

    def fail_commit(_source: Path, _target: Path) -> None:
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(dataset_module.os, "replace", fail_commit)
    with pytest.raises(OSError, match="interrupted"):
        collector.save(
            b"\0\0" * 100,
            16000,
            transcript="test",
            normalized_transcript="test",
            classification=None,
            classifier_status="disabled",
        )

    pending = tmp_path / "pending"
    assert not list(pending.glob("*_metadata.json"))
    assert "incomplete_write" in {issue.code for issue in scan_clips(pending).issues}


def test_backward_compatible_default_keeps_classifier_and_collection_disabled() -> None:
    config = AppConfig()
    assert not config.audio_classifier.enabled
    assert not config.training_data.enabled
    assert config.decision_fusion.classifier_primary


def test_metrics_and_confusion_matrix() -> None:
    expected = ["jet", "jet", "other", "other"]
    predicted = ["jet", "other", "jet", "other"]
    metrics = binary_metrics(expected, predicted, "jet")
    assert metrics["precision"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    matrix = confusion_matrix(expected, predicted, ["jet", "other"])
    assert matrix["jet"]["other"] == 1


def test_radio_augmentation_is_reproducible() -> None:
    pcm = (np.sin(np.linspace(0, 50, 3200)) * 5000).astype("<i2").tobytes()
    config = AudioAugmentationConfig(
        enabled=True,
        noise_probability=1,
        clipping_probability=1,
        dropout_probability=1,
        band_pass_probability=1,
        compression_probability=1,
        frequency_response_probability=1,
    )
    first = augment_pcm(pcm, config, np.random.default_rng(config.random_seed))
    second = augment_pcm(pcm, config, np.random.default_rng(config.random_seed))
    assert first == second
    assert first != pcm


def test_mock_classifier_has_no_cloud_or_token_side_effects() -> None:
    started = time.perf_counter()
    result = MockAudioIntentClassifier(classification()).classify(b"\0\0", 16000, context())
    assert result.model_version == "test-v1"
    assert time.perf_counter() - started < 1
