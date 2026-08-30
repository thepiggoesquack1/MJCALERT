from __future__ import annotations

import json
import time
import wave
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from mry_alert.audio_classifier.dataset import load_metadata, scan_clips
from mry_alert.audio_classifier.features import (
    LocalTrainableAudioClassifier,
    extract_audio_features,
)
from mry_alert.audio_classifier.models import ClassificationContext, DestinationLabel, IntentLabel
from mry_alert.config import AudioAugmentationConfig


def read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"Expected mono 16-bit PCM WAV: {path}")
        return source.readframes(source.getnframes()), source.getframerate()


def augment_pcm(
    pcm: bytes,
    config: AudioAugmentationConfig,
    rng: np.random.Generator,
    sample_rate: int = 16000,
) -> bytes:
    if not config.enabled:
        return pcm
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    gain = rng.uniform(*config.gain_range)
    samples *= gain
    if rng.random() < config.noise_probability:
        amplitude = rng.uniform(0, config.maximum_noise_amplitude)
        samples += rng.normal(0, amplitude, samples.size)
    if samples.size and rng.random() < config.dropout_probability:
        width = max(1, int(samples.size * rng.uniform(0.005, 0.03)))
        start = int(rng.integers(0, max(1, samples.size - width)))
        samples[start : start + width] = 0
    if rng.random() < config.clipping_probability:
        samples = np.clip(samples, -0.65, 0.65)
    if samples.size and rng.random() < config.band_pass_probability:
        spectrum = np.fft.rfft(samples)
        frequencies = np.fft.rfftfreq(samples.size, 1 / sample_rate)
        spectrum[
            (frequencies < config.band_pass_low_hz) | (frequencies > config.band_pass_high_hz)
        ] = 0
        samples = np.fft.irfft(spectrum, n=samples.size).astype(np.float32)
    if samples.size and rng.random() < config.frequency_response_probability:
        spectrum = np.fft.rfft(samples)
        tilt = np.linspace(rng.uniform(0.65, 1.0), rng.uniform(0.65, 1.0), spectrum.size)
        samples = np.fft.irfft(spectrum * tilt, n=samples.size).astype(np.float32)
    if rng.random() < config.compression_probability:
        # Coarse companding approximates low-bitrate radio codec artifacts.
        samples = np.round(np.sign(samples) * np.log1p(31 * np.abs(samples)) / np.log(32) * 63)
        samples = np.sign(samples) * np.expm1(np.abs(samples) / 63 * np.log(32)) / 31
    factor = float(rng.choice(config.speed_factors))
    if factor != 1 and samples.size > 2:
        old = np.arange(samples.size)
        new = np.arange(0, samples.size - 1, factor)
        samples = np.interp(new, old, samples)
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


def _reviewed_examples(
    dataset: Path,
) -> list[tuple[Path, str, str]]:
    examples: list[tuple[Path, str, str]] = []
    directories = [dataset]
    hard_negatives = dataset.parent / "hard_negatives"
    if hard_negatives != dataset and hard_negatives.exists():
        directories.append(hard_negatives)
    for directory in directories:
        scan = scan_clips(directory, require_reviewed=True)
        for metadata_path in scan.valid_metadata:
            metadata = load_metadata(metadata_path)
            assert metadata.reviewed_labels is not None
            assert metadata.wav_file is not None
            examples.append(
                (
                    metadata_path.parent / metadata.wav_file,
                    metadata.reviewed_labels.destination,
                    metadata.reviewed_labels.intent,
                )
            )
    return examples


def train_classifier(
    dataset: Path,
    output: Path,
    augmentation: AudioAugmentationConfig,
    *,
    minimum_examples_per_label: int = 1,
) -> dict[str, Any]:
    examples = _reviewed_examples(dataset)
    if not examples:
        raise ValueError("No human-reviewed WAV examples were found")
    rng = np.random.default_rng(augmentation.random_seed)
    heads: dict[str, dict[str, list[np.ndarray]]] = {
        "destination": defaultdict(list),
        "intent": defaultdict(list),
    }
    for wav_path, destination, intent in examples:
        DestinationLabel(destination)
        IntentLabel(intent)
        pcm, sample_rate = read_wav(wav_path)
        for value in (pcm, augment_pcm(pcm, augmentation, rng, sample_rate)):
            features = extract_audio_features(value, sample_rate)
            heads["destination"][destination].append(features)
            heads["intent"][intent].append(features)
    counts = {
        head: {label: len(values) // 2 for label, values in labels.items()}
        for head, labels in heads.items()
    }
    sparse = [
        f"{head}:{label}"
        for head, labels in counts.items()
        for label, count in labels.items()
        if count < minimum_examples_per_label
    ]
    if sparse:
        raise ValueError(f"Insufficient reviewed examples for {', '.join(sparse)}")
    centroids = {
        head: {
            label: np.mean(np.stack(values), axis=0).astype(float).tolist()
            for label, values in labels.items()
        }
        for head, labels in heads.items()
    }
    output.mkdir(parents=True, exist_ok=True)
    version = f"centroid-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    payload = {
        "format": "mry-audio-centroids-v1",
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "feature_bins": 32,
        "temperature": 0.25,
        "heads": centroids,
        "class_counts": counts,
        "class_weights": {
            head: {
                label: sum(labels.values()) / (len(labels) * count)
                for label, count in labels.items()
            }
            for head, labels in counts.items()
        },
        "training_examples": len(examples),
        "label_policy": "human-reviewed-only",
    }
    (output / "model.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def confusion_matrix(
    expected: list[str], predicted: list[str], labels: list[str]
) -> dict[str, dict[str, int]]:
    matrix = {actual: {guess: 0 for guess in labels} for actual in labels}
    for actual, guess in zip(expected, predicted, strict=True):
        matrix[actual][guess] += 1
    return matrix


def binary_metrics(expected: list[str], predicted: list[str], positive: str) -> dict[str, float]:
    counts = Counter(
        (
            "tp"
            if actual == positive and guess == positive
            else "fn"
            if actual == positive
            else "fp"
            if guess == positive
            else "tn"
        )
        for actual, guess in zip(expected, predicted, strict=True)
    )
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
    }


def evaluate_classifier(dataset: Path, model: Path) -> dict[str, Any]:
    examples = _reviewed_examples(dataset)
    if not examples:
        raise ValueError("No human-reviewed WAV examples were found")
    classifier = LocalTrainableAudioClassifier(model)
    expected_destinations: list[str] = []
    predicted_destinations: list[str] = []
    expected_intents: list[str] = []
    predicted_intents: list[str] = []
    confidences: list[float] = []
    durations: list[float] = []
    inference_times: list[float] = []
    for wav_path, destination, intent in examples:
        pcm, sample_rate = read_wav(wav_path)
        context = ClassificationContext(timestamp=datetime.now(UTC), source="offline_evaluation")
        started = time.perf_counter()
        result = classifier.classify(pcm, sample_rate, context)
        inference_times.append(time.perf_counter() - started)
        durations.append(len(pcm) / 2 / sample_rate)
        expected_destinations.append(destination)
        predicted_destinations.append(result.destination.value)
        expected_intents.append(intent)
        predicted_intents.append(result.intent.value)
        confidences.append(min(result.destination_confidence, result.intent_confidence))
    destination_labels = [label.value for label in DestinationLabel]
    intent_labels = [label.value for label in IntentLabel]
    destination_correct = [
        actual == guess
        for actual, guess in zip(expected_destinations, predicted_destinations, strict=True)
    ]
    calibration_error = sum(
        abs(float(correct) - confidence)
        for correct, confidence in zip(destination_correct, confidences, strict=True)
    ) / len(confidences)
    route_positive = IntentLabel.TAXI_OR_ROUTE.value
    correction_positive = IntentLabel.CORRECTION.value
    total_audio = sum(durations)
    total_inference = sum(inference_times)
    return {
        "model_version": classifier.model_version,
        "examples": len(examples),
        "destination_accuracy": sum(destination_correct) / len(examples),
        "jet_center": binary_metrics(
            expected_destinations,
            predicted_destinations,
            DestinationLabel.MONTEREY_JET_CENTER.value,
        ),
        "correction": binary_metrics(expected_intents, predicted_intents, correction_positive),
        "route_intent": binary_metrics(expected_intents, predicted_intents, route_positive),
        "destination_confusion_matrix": confusion_matrix(
            expected_destinations, predicted_destinations, destination_labels
        ),
        "intent_confusion_matrix": confusion_matrix(
            expected_intents, predicted_intents, intent_labels
        ),
        "confidence_calibration_error": calibration_error,
        "average_inference_seconds": total_inference / len(examples),
        "real_time_factor": total_inference / total_audio if total_audio else 0.0,
        "per_noise_level": {
            "unlabeled": {
                "examples": len(examples),
                "destination_accuracy": sum(destination_correct) / len(examples),
            }
        },
    }
