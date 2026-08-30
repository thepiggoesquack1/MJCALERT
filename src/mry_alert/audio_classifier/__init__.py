"""Local, optional ATC audio intent classification."""

from __future__ import annotations

import logging

from mry_alert.audio_classifier.base import (
    AudioIntentClassifier,
    DisabledAudioIntentClassifier,
)
from mry_alert.audio_classifier.features import (
    LocalTrainableAudioClassifier,
    RuleBasedFeatureClassifier,
)
from mry_alert.config import AppConfig

logger = logging.getLogger(__name__)


def create_audio_classifier(config: AppConfig) -> AudioIntentClassifier:
    if not config.audio_classifier.enabled:
        return DisabledAudioIntentClassifier()
    if config.audio_classifier.backend == "rule_based":
        return RuleBasedFeatureClassifier()
    if config.audio_classifier.backend == "local":
        return LocalTrainableAudioClassifier(config.audio_classifier.model_path)
    raise RuntimeError(
        "The mock audio classifier backend is available only through dependency injection"
    )


__all__ = ["AudioIntentClassifier", "create_audio_classifier"]
