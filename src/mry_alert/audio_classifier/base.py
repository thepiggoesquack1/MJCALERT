from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from mry_alert.audio_classifier.models import (
    AudioIntentResult,
    ClassificationContext,
)


class AudioIntentClassifier(Protocol):
    @property
    def model_version(self) -> str: ...

    def classify(
        self,
        pcm: bytes,
        sample_rate: int,
        context: ClassificationContext,
    ) -> AudioIntentResult: ...


class DisabledAudioIntentClassifier:
    model_version = "disabled"

    def classify(
        self,
        pcm: bytes,
        sample_rate: int,
        context: ClassificationContext,
    ) -> AudioIntentResult:
        del pcm, sample_rate, context
        return AudioIntentResult(reasons=["audio classifier disabled"])


class MockAudioIntentClassifier:
    def __init__(
        self,
        result: AudioIntentResult | Callable[[bytes], AudioIntentResult],
    ) -> None:
        self._result = result

    @property
    def model_version(self) -> str:
        if isinstance(self._result, AudioIntentResult):
            return self._result.model_version
        return "mock"

    def classify(
        self,
        pcm: bytes,
        sample_rate: int,
        context: ClassificationContext,
    ) -> AudioIntentResult:
        del sample_rate, context
        return self._result(pcm) if callable(self._result) else self._result
