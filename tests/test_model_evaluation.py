from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from mry_alert.adsb.mock import MockNearbyAircraftProvider
from mry_alert.cli import compare_models
from mry_alert.config import AppConfig, SpeechConfig, SpeechModelOverride
from mry_alert.server.audio_ingest import LiveAudioIngestService
from mry_alert.transcription.benchmark import benchmark_model, cleanup_transcriber
from mry_alert.transcription.benchmark_process import run_benchmark_subprocess
from mry_alert.transcription.evaluation import (
    aviation_entity_scores,
    character_error_rate,
    classify_rtf,
    evaluation_metrics,
    extract_aviation_entities,
    load_expected_json,
    load_expected_transcript,
    word_error_rate,
)
from mry_alert.transcription.faster_whisper import FasterWhisperTranscriber
from mry_alert.transcription.mock import MockTranscriber
from mry_alert.transcription.runtime import diagnose_speech_runtime


@pytest.mark.parametrize(
    "model",
    ["base.en", "small.en", "medium.en", "distil-large-v3"],
)
def test_standard_model_name_accepted(model: str) -> None:
    assert SpeechConfig(model=model).model == model


def test_hugging_face_model_identifier_accepted() -> None:
    model = "jacktol/whisper-medium.en-fine-tuned-for-ATC-faster-whisper"
    assert SpeechConfig(model=model).model == model


@pytest.mark.parametrize(
    "model", ["../secret", "owner/repo/extra", "https://example.test/x", "bad name"]
)
def test_invalid_model_identifier_rejected_safely(model: str) -> None:
    with pytest.raises(ValidationError):
        SpeechConfig(model=model)


class FakeCTranslate2:
    __version__ = "4.test"

    def __init__(self, devices: int) -> None:
        self.devices = devices

    def get_cuda_device_count(self) -> int:
        return self.devices

    def get_supported_compute_types(self, device: str) -> set[str]:
        assert device == "cuda"
        return {"float16", "int8_float16"}


class DiagnosticTranscriber:
    loaded_model = "small.en"

    def __init__(self, *_: object) -> None:
        pass


def test_cuda_diagnostic_output() -> None:
    config = AppConfig(speech=SpeechConfig(device="cuda", compute_type="float16"))
    result = diagnose_speech_runtime(
        config,
        ctranslate2_module=FakeCTranslate2(1),
        transcriber_factory=DiagnosticTranscriber,  # type: ignore[arg-type]
    )
    assert result.cuda_available and result.initialized
    assert "float16" in result.supported_cuda_compute_types


def test_cpu_diagnostic_output() -> None:
    result = diagnose_speech_runtime(
        AppConfig(),
        ctranslate2_module=FakeCTranslate2(0),
        transcriber_factory=DiagnosticTranscriber,  # type: ignore[arg-type]
    )
    assert result.initialized and not result.cuda_available


def test_cuda_diagnostic_explains_missing_runtime() -> None:
    config = AppConfig(speech=SpeechConfig(device="cuda"))
    result = diagnose_speech_runtime(config, ctranslate2_module=FakeCTranslate2(0))
    assert not result.initialized
    assert result.error and "CUDA/cuDNN" in result.error


def test_model_initialization_failure_has_no_silent_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail(*_: object, **__: object) -> object:
        raise OSError("missing model")

    with pytest.raises(RuntimeError), caplog.at_level(logging.ERROR):
        FasterWhisperTranscriber(SpeechConfig(model="small.en"), model_factory=fail)
    assert "INITIALIZATION FAILED" in caplog.text
    assert "Fallback model" not in caplog.text


def test_explicit_fallback_behavior(caplog: pytest.LogCaptureFixture) -> None:
    calls: list[str] = []

    def factory(model: str, **_: object) -> object:
        calls.append(model)
        if model == "owner/atc-model":
            raise OSError("incompatible CUDA")
        return object()

    config = SpeechConfig(
        model="owner/atc-model", fallback_model="base.en", allow_model_fallback=True
    )
    with caplog.at_level(logging.WARNING):
        transcriber = FasterWhisperTranscriber(config, model_factory=factory)
    assert calls == ["owner/atc-model", "base.en"]
    assert transcriber.loaded_model == "base.en"
    assert "Fallback model: base.en" in caplog.text


def test_per_model_overrides() -> None:
    config = AppConfig(
        speech_model_overrides={"base.en": SpeechModelOverride(beam_size=3, compute_type="int8")}
    )
    selected = config.speech_for_model("base.en")
    assert selected.beam_size == 3 and selected.compute_type == "int8"


def test_invalid_override_key_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig(speech_model_overrides={"../bad": SpeechModelOverride()})


def test_error_rates() -> None:
    assert word_error_rate("taxi alpha", "taxi bravo") == 0.5
    assert character_error_rate("abc", "adc") == pytest.approx(1 / 3)


def test_aviation_entity_scoring() -> None:
    config = AppConfig()
    reference = extract_aviation_entities(
        "November eight two five Sierra Papa taxi via Alpha to Monterey Jet Center",
        config.destinations,
    )
    actual = extract_aviation_entities(
        "November eight two five Sierra Papa taxi via Alpha to Monterey Jet Center",
        config.destinations,
    )
    scores = aviation_entity_scores(reference, actual)
    assert all(value == 1 for value in scores.values())


def test_evaluation_metrics_include_route_and_entities() -> None:
    metrics = evaluation_metrics(
        "taxi via alpha to Monterey Jet Center",
        "taxi via alpha to Monterey Jet Center",
        AppConfig().destinations,
    )
    assert metrics["word_error_rate"] == 0
    assert metrics["destination_accuracy"] == 1
    assert metrics["route_taxiway_term_accuracy"] == 1


@pytest.mark.parametrize(
    ("rtf", "expected"),
    [(0.5, "excellent"), (0.7, "acceptable"), (0.9, "risky"), (1.01, "unsuitable for live use")],
)
def test_rtf_classification(rtf: float, expected: str) -> None:
    assert classify_rtf(rtf) == expected


def test_expected_files_load(tmp_path: Path) -> None:
    transcript = tmp_path / "expected.txt"
    transcript.write_text("Monterey Jet Center", encoding="utf-8")
    metadata = tmp_path / "expected.json"
    metadata.write_text('{"should_notify": true}', encoding="utf-8")
    assert load_expected_transcript(transcript) == "Monterey Jet Center"
    assert load_expected_json(metadata)["should_notify"] is True


@pytest.mark.asyncio
async def test_comparison_output_and_detector_integration() -> None:
    def factory(*_: object) -> MockTranscriber:
        return MockTranscriber(
            ["November eight two five Sierra Papa request taxi to Monterey Jet Center"]
        )

    report = await benchmark_model(
        "base.en",
        [b"\0\0" * 16000],
        1.0,
        AppConfig(),
        "November eight two five Sierra Papa request taxi to Monterey Jet Center",
        transcriber_factory=factory,
    )
    assert report["model"] == "base.en"
    assert report["segments"][0]["final_detector_decision"] == "PENDING"
    assert report["segments"][0]["notification_would_be_sent"] is True
    assert report["notifications_sent"] == 0
    assert report["expected_transcript_metrics"]["word_error_rate"] == 0


def test_tokens_are_never_part_of_model_logs(caplog: pytest.LogCaptureFixture) -> None:
    secret_pairing = "pairing-super-secret"
    secret_hf = "hf_super_secret"

    def fail(*_: object, **__: object) -> object:
        raise OSError("model unavailable")

    config = SpeechConfig(model="owner/model")
    with pytest.raises(RuntimeError), caplog.at_level(logging.INFO):
        FasterWhisperTranscriber(config, model_factory=fail)
    assert secret_pairing not in caplog.text
    assert secret_hf not in caplog.text


@pytest.mark.asyncio
async def test_live_backlog_warning(caplog: pytest.LogCaptureFixture) -> None:
    class Socket:
        async def send_json(self, value: dict[str, Any]) -> None:
            self.value = value

    async def publish(_: object) -> None:
        return None

    service = LiveAudioIngestService(
        AppConfig(),
        publish,  # type: ignore[arg-type]
        transcriber=MockTranscriber(["ambiguous transmission"]),
        provider=MockNearbyAircraftProvider([]),
    )
    with (
        patch("mry_alert.server.audio_ingest.time.perf_counter", side_effect=[0.0, 1.0]),
        caplog.at_level(logging.WARNING),
    ):
        await service._process_transmission(
            Socket(),  # type: ignore[arg-type]
            MockTranscriber(["ambiguous transmission"]),
            b"\0\0" * 16000,
            queue_depth=4,
        )
    assert "LIVE TRANSCRIPTION WARNING" in caplog.text
    assert "Queued segments: 4" in caplog.text


class FakeProcess:
    def __init__(self, command: list[str], outcome: str) -> None:
        self.command = command
        self.outcome = outcome
        self.returncode = 0
        self.terminated = False
        self.killed = False

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        if self.outcome == "timeout" and not self.terminated:
            raise subprocess.TimeoutExpired(self.command, timeout or 0)
        if self.outcome == "interrupt" and not self.terminated:
            raise KeyboardInterrupt
        if self.outcome == "failed":
            self.returncode = 1
            return "", "model exception"
        output = Path(self.command[self.command.index("--output") + 1])
        output.write_text(
            json.dumps({"model": "base.en", "status": "success", "segments": []}),
            encoding="utf-8",
        )
        return "worker complete", ""

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0


@pytest.mark.parametrize(
    ("outcome", "status"),
    [("success", "success"), ("timeout", "timed_out"), ("failed", "failed")],
)
def test_benchmark_subprocess_status(outcome: str, status: str) -> None:
    process: FakeProcess | None = None

    def factory(command: list[str], **_: object) -> FakeProcess:
        nonlocal process
        process = FakeProcess(command, outcome)
        return process

    result = run_benchmark_subprocess(
        {"model": "base.en"},
        0.01,
        popen_factory=factory,  # type: ignore[arg-type]
    )
    assert result["status"] == status
    if outcome == "timeout":
        assert process and process.terminated


def test_benchmark_subprocess_clean_interrupt() -> None:
    process: FakeProcess | None = None

    def factory(command: list[str], **_: object) -> FakeProcess:
        nonlocal process
        process = FakeProcess(command, "interrupt")
        return process

    result = run_benchmark_subprocess(
        {"model": "base.en"},
        10,
        popen_factory=factory,  # type: ignore[arg-type]
    )
    assert result["status"] == "interrupted"
    assert process and process.terminated


def test_model_cleanup_calls_supported_close() -> None:
    class Model:
        closed = False

        def close(self) -> None:
            self.closed = True

    transcriber = SimpleNamespace(model=Model())
    cleanup_transcriber(transcriber)
    assert transcriber.model.closed


@pytest.mark.asyncio
async def test_comparison_continues_after_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def load(_: Path, __: AppConfig) -> tuple[float, list[bytes]]:
        return 1.0, [b"\0\0"]

    statuses = iter(["timed_out", "success"])

    def run(request: dict[str, Any], _: float) -> dict[str, Any]:
        status = next(statuses)
        if status == "timed_out":
            return {"model": request["model"], "status": status, "segments": []}
        return {
            "model": request["model"],
            "status": "success",
            "segments": [],
            "real_time_factor": 0.1,
            "performance_classification": "excellent",
        }

    monkeypatch.setattr("mry_alert.cli.load_segmented_wave", load)
    monkeypatch.setattr("mry_alert.cli.run_benchmark_subprocess", run)
    assert (
        await compare_models(Path("fake.wav"), ["base.en", "small.en"], AppConfig(), None, False)
        == 0
    )
