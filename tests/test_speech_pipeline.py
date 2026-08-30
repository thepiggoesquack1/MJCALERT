from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mry_alert.audio.preprocessing import preprocess_pcm16
from mry_alert.audio.segmenter import VadSegmenter
from mry_alert.config import AppConfig, AudioConfig, AudioPreprocessingConfig, SpeechConfig
from mry_alert.models import NearbyAircraft
from mry_alert.transcription.adsb_prompt import build_adsb_prompt, registration_spoken_form
from mry_alert.transcription.faster_whisper import FasterWhisperTranscriber


class FakeModel:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def transcribe(self, _audio: np.ndarray, **kwargs: Any):
        self.kwargs = kwargs
        segment = SimpleNamespace(
            text=" November eight two five sierra papa ",
            avg_logprob=-0.3,
            no_speech_prob=0.02,
            end=1.2,
        )
        return iter([segment]), SimpleNamespace(language_probability=0.99)


def pcm(values: np.ndarray) -> bytes:
    return values.astype(np.int16).tobytes()


def test_new_speech_and_segmentation_defaults() -> None:
    config = AppConfig()
    assert config.speech.model == "small.en"
    assert config.speech.language == "en"
    assert "Monterey Ground communications at KMRY" in config.speech.initial_prompt
    assert config.audio.pre_roll_ms == 250
    assert config.audio.end_silence_ms == 350
    assert config.audio.max_transmission_seconds == 8


def test_decoding_options_and_dynamic_prompt_are_forwarded() -> None:
    model = FakeModel()
    config = SpeechConfig(initial_prompt="Static KMRY prompt")
    transcriber = FasterWhisperTranscriber(
        config, AudioPreprocessingConfig(enabled=False), model=model
    )
    transcriber.set_dynamic_prompt("Nearby aircraft registrations: N825SP.")
    result = transcriber.transcribe(pcm(np.zeros(160, dtype=np.int16)), 16000)
    assert result.text == "November eight two five sierra papa"
    assert model.kwargs["language"] == "en"
    assert model.kwargs["beam_size"] == 5
    assert model.kwargs["temperature"] == 0.0
    assert model.kwargs["condition_on_previous_text"] is False
    assert model.kwargs["vad_filter"] is False
    assert model.kwargs["initial_prompt"].startswith("Static KMRY prompt")
    assert "N825SP" in model.kwargs["initial_prompt"]


def test_configuration_overrides_are_forwarded() -> None:
    model = FakeModel()
    config = SpeechConfig(
        model="medium.en",
        beam_size=3,
        temperature=0.2,
        condition_on_previous_text=True,
        vad_filter=True,
    )
    FasterWhisperTranscriber(
        config, AudioPreprocessingConfig(enabled=False), model=model
    ).transcribe(pcm(np.zeros(20, dtype=np.int16)), 16000)
    assert model.kwargs["beam_size"] == 3
    assert model.kwargs["temperature"] == 0.2
    assert model.kwargs["condition_on_previous_text"] is True
    assert model.kwargs["vad_filter"] is True


def test_preprocessing_silence_clipping_low_level_and_disabled() -> None:
    config = AudioPreprocessingConfig()
    silence = pcm(np.zeros(1000, dtype=np.int16))
    assert preprocess_pcm16(silence, 16000, config) == silence
    clipped = pcm(np.array([32767, -32768] * 500, dtype=np.int16))
    processed = np.frombuffer(preprocess_pcm16(clipped, 16000, config), dtype=np.int16)
    assert np.max(np.abs(processed.astype(np.int32))) <= round(0.98 * 32767)
    low = pcm((np.sin(np.linspace(0, 20, 2000)) * 100).astype(np.int16))
    assert np.max(np.abs(np.frombuffer(preprocess_pcm16(low, 16000, config), dtype=np.int16))) > 100
    assert preprocess_pcm16(clipped, 16000, AudioPreprocessingConfig(enabled=False)) == clipped
    normal = pcm((np.sin(np.linspace(0, 100, 4000)) * 8000).astype(np.int16))
    assert len(preprocess_pcm16(normal, 16000, config)) == len(normal)
    with pytest.raises(ValueError, match="expects 16000 Hz"):
        preprocess_pcm16(normal, 8000, config)


def test_adsb_prompt_spoken_forms_and_limit() -> None:
    assert registration_spoken_form("N825SP") == "November Eight Two Five Sierra Papa"
    aircraft = [
        NearbyAircraft(hex=str(index), registration=registration, distance_nm=float(index))
        for index, registration in enumerate(["N825SP", "N123AB", "N441QS"])
    ]
    prompt = build_adsb_prompt(aircraft, maximum=2)
    assert "N825SP" in prompt and "N123AB" in prompt
    assert "N441QS" not in prompt
    assert "November Eight Two Five Sierra Papa" in prompt
    assert build_adsb_prompt([], maximum=5) == ""


class SequenceVad:
    def __init__(self, values: list[bool]) -> None:
        self.values = iter(values)

    def is_speech(self, _frame: bytes, _sample_rate: int) -> bool:
        return next(self.values)


def test_vad_preserves_preroll_and_ends_after_silence() -> None:
    config = AudioConfig(pre_roll_ms=250, end_silence_ms=350)
    values = [False] * 8 + [True] + [False] * 12
    segmenter = VadSegmenter(config, SequenceVad(values))
    frame = b"\0" * 960
    outputs = [segmenter.add_frame(frame) for _ in values]
    transmission = next(item for item in outputs if item is not None)
    assert len(transmission) == 21 * len(frame)


def test_vad_splits_at_eight_second_limit() -> None:
    config = AudioConfig(frame_duration_ms=30, max_transmission_seconds=8)
    maximum = int(8000 / 30)
    segmenter = VadSegmenter(config, SequenceVad([True] * maximum))
    frame = b"\1" * 960
    outputs = [segmenter.add_frame(frame) for _ in range(maximum)]
    assert outputs[-1] is not None
    assert len(outputs[-1] or b"") == maximum * len(frame)
