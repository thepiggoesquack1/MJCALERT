from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from mry_alert.audio_classifier.models import (
    AudioIntentResult,
    ClassificationContext,
    DestinationLabel,
    IntentLabel,
)

FEATURE_BINS = 32


def extract_audio_features(pcm: bytes, sample_rate: int) -> np.ndarray:
    """Extract deterministic, lightweight radio-audio features without network access."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size == 0:
        return np.zeros(FEATURE_BINS + 6, dtype=np.float32)
    frame = max(128, int(sample_rate * 0.025))
    hop = max(64, int(sample_rate * 0.010))
    if samples.size < frame:
        samples = np.pad(samples, (0, frame - samples.size))
    window = np.hanning(frame).astype(np.float32)
    spectra: list[np.ndarray] = []
    for start in range(0, samples.size - frame + 1, hop):
        power = np.abs(np.fft.rfft(samples[start : start + frame] * window)) ** 2
        bands = np.array_split(np.log1p(power), FEATURE_BINS)
        spectra.append(np.array([float(np.mean(band)) for band in bands]))
    spectral = np.mean(np.stack(spectra), axis=0)
    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(np.max(np.abs(samples)))
    zero_crossing = float(np.mean(np.abs(np.diff(np.signbit(samples)))))
    clipping = float(np.mean(np.abs(samples) >= 0.98))
    duration = float(samples.size / sample_rate)
    voiced_fraction = float(np.mean(np.abs(samples) > max(0.01, rms * 0.5)))
    values = np.concatenate(
        [spectral, np.array([rms, peak, zero_crossing, clipping, duration, voiced_fraction])]
    ).astype(np.float32)
    norm = float(np.linalg.norm(values))
    return values / norm if norm else values


class RuleBasedFeatureClassifier:
    """Conservative disabled-model fallback; it can reject noise but never assert semantics."""

    model_version = "rule-features-v1"

    def classify(
        self,
        pcm: bytes,
        sample_rate: int,
        context: ClassificationContext,
    ) -> AudioIntentResult:
        del context
        started = time.perf_counter()
        features = extract_audio_features(pcm, sample_rate)
        rms = float(features[-6])
        duration = len(pcm) / 2 / sample_rate if sample_rate else 0
        noise = 0.95 if not pcm or duration < 0.12 or rms < 0.001 else 0.15
        intent = IntentLabel.NOISE if noise >= 0.65 else IntentLabel.NONE
        return AudioIntentResult(
            intent=intent,
            intent_confidence=noise if intent == IntentLabel.NOISE else 0.0,
            noise_confidence=noise,
            model_version=self.model_version,
            reasons=[
                "feature fallback only rejects unusable audio",
                "fallback never infers destination semantics",
            ],
            inference_seconds=time.perf_counter() - started,
        )


class LocalTrainableAudioClassifier:
    """Small nearest-centroid multi-head baseline stored as portable JSON."""

    def __init__(self, model_path: Path) -> None:
        path = model_path / "model.json" if model_path.is_dir() else model_path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Unable to load audio classifier model {path}: {exc}") from exc
        if payload.get("format") != "mry-audio-centroids-v1":
            raise RuntimeError(f"Unsupported audio classifier model format: {path}")
        self._payload = payload
        self._temperature = max(float(payload.get("temperature", 0.25)), 0.01)
        self.model_version = str(payload.get("version", "local-baseline"))

    def _predict(self, features: np.ndarray, head: str) -> tuple[str, float, dict[str, float]]:
        centroids = self._payload.get("heads", {}).get(head, {})
        if not centroids:
            raise RuntimeError(f"Audio classifier model has no {head} head")
        labels = list(centroids)
        distances = np.array(
            [
                float(np.linalg.norm(features - np.asarray(centroids[label], dtype=np.float32)))
                for label in labels
            ]
        )
        logits = -distances / self._temperature
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        probabilities /= float(np.sum(probabilities))
        index = int(np.argmax(probabilities))
        scores = {label: float(probabilities[i]) for i, label in enumerate(labels)}
        return labels[index], float(probabilities[index]), scores

    def classify(
        self,
        pcm: bytes,
        sample_rate: int,
        context: ClassificationContext,
    ) -> AudioIntentResult:
        del context
        started = time.perf_counter()
        features = extract_audio_features(pcm, sample_rate)
        destination, destination_confidence, destination_scores = self._predict(
            features, "destination"
        )
        intent, intent_confidence, intent_scores = self._predict(features, "intent")
        correction_confidence = intent_scores.get(IntentLabel.CORRECTION.value, 0.0)
        noise_confidence = intent_scores.get(IntentLabel.NOISE.value, 0.0)
        return AudioIntentResult(
            destination=DestinationLabel(destination),
            destination_confidence=destination_confidence,
            intent=IntentLabel(intent),
            intent_confidence=intent_confidence,
            correction=correction_confidence >= 0.5,
            correction_confidence=correction_confidence,
            noise_confidence=noise_confidence,
            raw_scores={
                **{f"destination:{key}": value for key, value in destination_scores.items()},
                **{f"intent:{key}": value for key, value in intent_scores.items()},
            },
            model_version=self.model_version,
            reasons=["local nearest-centroid acoustic baseline"],
            inference_seconds=time.perf_counter() - started,
        )
