from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language_probability: float | None = None
    average_log_probability: float | None = None
    no_speech_probability: float | None = None
    segment_count: int | None = None
    duration_seconds: float | None = None
    quality: str | None = None


class Transcriber(Protocol):
    def transcribe(self, pcm: bytes, sample_rate: int) -> TranscriptionResult: ...
