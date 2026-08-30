from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from mry_alert.audio.preprocessing import preprocess_pcm16
from mry_alert.config import AudioPreprocessingConfig, SpeechConfig
from mry_alert.transcription.base import TranscriptionResult

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber:
    def __init__(
        self,
        config: SpeechConfig,
        preprocessing: AudioPreprocessingConfig | None = None,
        model: Any | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.preprocessing = preprocessing or AudioPreprocessingConfig()
        self._dynamic_prompt = ""
        if self.preprocessing.noise_reduction:
            logger.warning(
                "Audio noise reduction requested but no lightweight noise-reduction dependency "
                "is installed; filtering, normalization, and limiting remain enabled"
            )
        self.requested_model = config.model
        self.loaded_model = config.model
        self.initialization_seconds = 0.0
        if model is None:
            if model_factory is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise RuntimeError(
                        "Local speech support is not installed. Run: pip install -e .[speech]"
                    ) from exc
                factory = WhisperModel
            else:
                factory = model_factory
            logger.info(
                "SPEECH MODEL\nModel: %s\nDevice: %s\nCompute type: %s\n"
                "Status: downloading/cache resolution (download occurs only when uncached)",
                config.model,
                config.device,
                config.compute_type,
            )
            logger.info(
                "SPEECH MODEL\nModel: %s\nDevice: %s\nCompute type: %s\nStatus: loading",
                config.model,
                config.device,
                config.compute_type,
            )
            started = time.perf_counter()
            try:
                model = factory(
                    config.model, device=config.device, compute_type=config.compute_type
                )
            except Exception as requested_error:
                fallback = config.fallback_model
                if not config.allow_model_fallback or not fallback or fallback == config.model:
                    logger.error(
                        "SPEECH MODEL INITIALIZATION FAILED\nRequested model: %s\nReason: %s",
                        config.model,
                        requested_error,
                    )
                    raise RuntimeError(
                        f"Speech model {config.model!r} failed to initialize: {requested_error}"
                    ) from requested_error
                logger.warning(
                    "SPEECH MODEL INITIALIZATION FAILED\nRequested model: %s\n\n"
                    "Fallback model: %s\n\nReason: %s",
                    config.model,
                    fallback,
                    requested_error,
                )
                try:
                    model = factory(
                        fallback, device=config.device, compute_type=config.compute_type
                    )
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"Speech models {config.model!r} and fallback {fallback!r} failed "
                        f"to initialize: {fallback_error}"
                    ) from fallback_error
                self.loaded_model = fallback
            self.initialization_seconds = time.perf_counter() - started
            logger.info(
                "SPEECH MODEL\nModel: %s\nDevice: %s\nCompute type: %s\n"
                "Status: ready\nInitialization seconds: %.3f",
                self.loaded_model,
                config.device,
                config.compute_type,
                self.initialization_seconds,
            )
        self.model = model

    def set_dynamic_prompt(self, prompt: str) -> None:
        self._dynamic_prompt = prompt

    def transcribe(self, pcm: bytes, sample_rate: int) -> TranscriptionResult:
        if sample_rate != 16000:
            raise ValueError("FasterWhisperTranscriber expects 16 kHz PCM")
        preprocessing_started = time.perf_counter()
        processed = preprocess_pcm16(pcm, sample_rate, self.preprocessing)
        logger.info(
            "Speech timing: preprocessing completed in %.3f seconds",
            time.perf_counter() - preprocessing_started,
        )
        audio = np.frombuffer(processed, dtype=np.int16).astype(np.float32) / 32768.0
        prompt = self.config.initial_prompt if self.config.use_static_aviation_prompt else ""
        if self._dynamic_prompt and self.config.use_adsb_dynamic_prompt:
            prompt = f"{prompt.rstrip()}\n{self._dynamic_prompt}"
        logger.info("Speech timing: transcribe() entry for %.3f seconds of audio", len(pcm) / 32000)
        try:
            segments, info = self.model.transcribe(
                audio,
                language=self.config.language,
                initial_prompt=prompt or None,
                beam_size=self.config.beam_size,
                temperature=self.config.temperature,
                condition_on_previous_text=self.config.condition_on_previous_text,
                vad_filter=self.config.vad_filter,
            )
        except TypeError as exc:
            raise RuntimeError(
                "Installed faster-whisper does not support the configured decoding options. "
                "Upgrade faster-whisper or adjust the speech configuration."
            ) from exc
        logger.info("Speech timing: consuming faster-whisper segment iterator")
        iterator_started = time.perf_counter()
        segment_list = list(segments)
        logger.info(
            "Speech timing: segment iterator consumed exactly once (%d segments) in %.3f seconds",
            len(segment_list),
            time.perf_counter() - iterator_started,
        )
        text = " ".join(segment.text.strip() for segment in segment_list).strip()
        probability = getattr(info, "language_probability", None)
        log_values = [
            float(value)
            for segment in segment_list
            if (value := getattr(segment, "avg_logprob", None)) is not None
        ]
        no_speech_values = [
            float(value)
            for segment in segment_list
            if (value := getattr(segment, "no_speech_prob", None)) is not None
        ]
        average_log_probability = sum(log_values) / len(log_values) if log_values else None
        no_speech_probability = (
            sum(no_speech_values) / len(no_speech_values) if no_speech_values else None
        )
        if average_log_probability is None:
            quality = None
        elif average_log_probability >= -0.55 and (no_speech_probability or 0) < 0.25:
            quality = "high"
        elif average_log_probability >= -1.15 and (no_speech_probability or 0) < 0.6:
            quality = "medium"
        else:
            quality = "low"
        duration = max((float(getattr(segment, "end", 0)) for segment in segment_list), default=0)
        return TranscriptionResult(
            text=text,
            language_probability=float(probability) if probability is not None else None,
            average_log_probability=average_log_probability,
            no_speech_probability=no_speech_probability,
            segment_count=len(segment_list),
            duration_seconds=duration,
            quality=quality,
        )
