from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any

from mry_alert.config import AppConfig
from mry_alert.transcription.faster_whisper import FasterWhisperTranscriber


@dataclass(frozen=True)
class SpeechRuntimeDiagnostic:
    model: str
    device: str
    compute_type: str
    ctranslate2_version: str | None
    cuda_available: bool
    supported_cuda_compute_types: list[str]
    initialized: bool
    initialization_seconds: float | None
    loaded_model: str | None
    approximate_gpu_memory_mb: float | None
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _gpu_memory_mb(module: Any) -> float | None:
    try:
        value = module.get_cuda_allocated_memory()
    except (AttributeError, RuntimeError, OSError):
        return None
    return round(float(value) / (1024 * 1024), 1)


def diagnose_speech_runtime(
    config: AppConfig,
    *,
    ctranslate2_module: Any | None = None,
    transcriber_factory: Callable[..., FasterWhisperTranscriber] = FasterWhisperTranscriber,
) -> SpeechRuntimeDiagnostic:
    speech = config.speech_for_model(config.speech.model)
    try:
        module: Any = ctranslate2_module or import_module("ctranslate2")
        version = str(getattr(module, "__version__", "unknown"))
        try:
            cuda_count = int(module.get_cuda_device_count())
        except (AttributeError, RuntimeError, OSError):
            cuda_count = 0
        cuda_available = cuda_count > 0
        try:
            supported = sorted(module.get_supported_compute_types("cuda"))
        except (AttributeError, RuntimeError, OSError):
            supported = []
    except ImportError as exc:
        return SpeechRuntimeDiagnostic(
            speech.model,
            speech.device,
            speech.compute_type,
            None,
            False,
            [],
            False,
            None,
            None,
            None,
            f"CTranslate2 is unavailable: {exc}. Install the local speech extra.",
        )
    if speech.device == "cuda" and not cuda_available:
        return SpeechRuntimeDiagnostic(
            speech.model,
            speech.device,
            speech.compute_type,
            version,
            False,
            supported,
            False,
            None,
            None,
            None,
            "CUDA was requested but CTranslate2 found no CUDA device. Check the NVIDIA "
            "driver and compatible CUDA/cuDNN libraries.",
        )
    started = time.perf_counter()
    try:
        transcriber = transcriber_factory(speech, config.audio.preprocessing)
    except (RuntimeError, OSError, ValueError) as exc:
        return SpeechRuntimeDiagnostic(
            speech.model,
            speech.device,
            speech.compute_type,
            version,
            cuda_available,
            supported,
            False,
            time.perf_counter() - started,
            None,
            _gpu_memory_mb(module),
            str(exc),
        )
    return SpeechRuntimeDiagnostic(
        speech.model,
        speech.device,
        speech.compute_type,
        version,
        cuda_available,
        supported,
        True,
        time.perf_counter() - started,
        transcriber.loaded_model,
        _gpu_memory_mb(module),
        None,
    )
