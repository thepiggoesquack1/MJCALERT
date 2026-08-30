from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from mry_alert.config import AppConfig
from mry_alert.control.core import (
    AppPaths,
    IndicatorState,
    ProcessOwnership,
    backend_command,
    display_command,
    fetch_json,
    filter_session_events,
    manual_command,
    pairing_token_path,
    parse_safe_command,
    post_test_alert,
    redact_secrets,
    resolve_app_paths,
    status_indicators,
    update_alert_sensitivity,
    update_training_collection,
    validate_config,
)


def test_event_history_search_filters_and_sorting() -> None:
    records = [
        {
            "event_id": "newer",
            "timestamp": "2026-08-05T18:01:00+00:00",
            "registration": "N627S",
            "aircraft_type_name": "Cessna Citation CJ3",
            "transcript_excerpt": "going to Monterey Jet Center",
            "transition_type": "confirmed",
            "final_decision": "confirmed",
        },
        {
            "event_id": "older",
            "timestamp": "2026-08-05T18:00:00+00:00",
            "registration": "N100J",
            "aircraft_type_name": "Pilatus PC-24",
            "transcript_excerpt": "at the Jet Center request taxi",
            "transition_type": "outbound_filtered",
            "final_decision": "ignored",
            "decision_reasons": ["outbound departure"],
        },
    ]

    assert [item["event_id"] for item in filter_session_events(records, "N627S")] == [
        "newer"
    ]
    assert filter_session_events(records, "Cessna")[0]["event_id"] == "newer"
    assert filter_session_events(records, "request taxi")[0]["event_id"] == "older"
    assert (
        filter_session_events(records, category="Outbound filtered")[0]["event_id"]
        == "older"
    )
    assert [
        item["event_id"]
        for item in filter_session_events(records, sort_order="Oldest first")
    ] == ["older", "newer"]


def paths(tmp_path: Path, backend: Path | None = None) -> AppPaths:
    return AppPaths(
        application_dir=tmp_path,
        config_path=tmp_path / "config.yaml",
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        gui_logs_dir=tmp_path / "logs/gui",
        backend_executable=backend,
    )


def test_alert_sensitivity_update_preserves_other_config_text(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "server:\n  port: 9876 # keep me\ndetection:\n  alert_threshold: 0.83\n",
        encoding="utf-8",
    )

    update_alert_sensitivity(config, "balanced")

    text = config.read_text(encoding="utf-8")
    loaded = AppConfig.load(config)
    assert "port: 9876 # keep me" in text
    assert "alert_threshold: 0.83" in text
    assert loaded.detection.alert_sensitivity == "balanced"


def test_alert_sensitivity_update_rejects_unknown_mode(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported alert sensitivity"):
        update_alert_sensitivity(config, "maximum")


def test_training_collection_update_preserves_other_settings_and_comment(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "server:\n  port: 9876\ntraining_data:\n"
        "  enabled: false # keep collection explicit\n"
        "  directory: custom/clips\n  save_audio: true\n",
        encoding="utf-8",
    )

    update_training_collection(config, True)

    text = config.read_text(encoding="utf-8")
    loaded = AppConfig.load(config)
    assert "enabled: true # keep collection explicit" in text
    assert "directory: custom/clips" in text
    assert "port: 9876" in text
    assert loaded.training_data.enabled is True


def test_backend_start_command_construction_source(tmp_path: Path) -> None:
    program, arguments = backend_command(paths(tmp_path), tmp_path / "config.yaml")
    assert arguments[:3] == ["-m", "mry_alert", "serve"]
    assert "--config" in arguments
    assert program


def test_backend_start_command_construction_bundled(tmp_path: Path) -> None:
    backend = tmp_path / "MRY Alert Backend.exe"
    program, arguments = backend_command(paths(tmp_path, backend), tmp_path / "config.yaml")
    assert program == str(backend)
    assert arguments == ["serve", "--config", str(tmp_path / "config.yaml")]


def test_safe_manual_command_parsing_preserves_windows_path() -> None:
    parsed = parse_safe_command(
        'replay --audio "C:\\ATC files\\sample.wav" --model base.en --config config.yaml'
    )
    assert parsed[0] == "replay"
    assert parsed[2] == "C:\\ATC files\\sample.wav"


@pytest.mark.parametrize(
    "command",
    [
        "replay --audio x.wav & whoami",
        "replay --audio x.wav && dir",
        "replay | more",
        "replay > output.txt",
        "replay; dir",
        "replay `whoami`",
        "replay $(whoami)",
    ],
)
def test_shell_operators_rejected(command: str) -> None:
    with pytest.raises(ValueError, match="Shell operators"):
        parse_safe_command(command)


def test_unsupported_command_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        parse_safe_command("python evil.py")


def test_manual_command_never_uses_shell(tmp_path: Path) -> None:
    program, arguments = manual_command(paths(tmp_path), ["list-audio-devices"])
    shown = display_command(program, arguments)
    assert "mry_alert" in shown
    assert "|" not in arguments


def test_process_start_stop_ownership() -> None:
    ownership = ProcessOwnership()
    assert ownership.can_start and not ownership.can_stop
    ownership.child_started()
    assert ownership.can_stop and not ownership.can_start
    ownership.child_stopped()
    assert ownership.can_start


def test_external_backend_cannot_be_stopped() -> None:
    ownership = ProcessOwnership()
    ownership.observe_health(True)
    assert ownership.external_detected
    assert not ownership.can_start
    assert not ownership.can_stop


def test_external_backend_prevents_child_start() -> None:
    ownership = ProcessOwnership(external_detected=True)
    with pytest.raises(RuntimeError):
        ownership.child_started()


def test_status_extension_connected() -> None:
    indicators = status_indicators(
        {"extension_clients": 1},
        process_running=True,
        health_ok=True,
        external=False,
        config_ok=True,
        token_found=True,
    )
    assert indicators["Chrome extension"].state == IndicatorState.HEALTHY


def test_audio_connected_but_inactive_is_yellow() -> None:
    indicators = status_indicators(
        {"audio_connected": True, "audio_recently_received": False},
        process_running=True,
        health_ok=True,
        external=False,
        config_ok=True,
        token_found=True,
    )
    assert indicators["Audio stream activity"].state == IndicatorState.DEGRADED
    assert "inactive" in indicators["Audio stream activity"].text


def test_adsb_healthy_state() -> None:
    indicators = status_indicators(
        {"adsb_ok": True, "adsb_tracked_aircraft": 6, "adsb_provider": "adsb_lol"},
        process_running=True,
        health_ok=True,
        external=False,
        config_ok=True,
        token_found=True,
    )
    assert indicators["ADS-B provider"].state == IndicatorState.HEALTHY


def test_adsb_stale_or_failed_state() -> None:
    indicators = status_indicators(
        {"adsb_ok": False, "adsb_error": "stale provider response"},
        process_running=True,
        health_ok=True,
        external=False,
        config_ok=True,
        token_found=True,
    )
    assert indicators["ADS-B provider"].state == IndicatorState.FAILED


def test_adsb_server_error_has_friendly_retry_message() -> None:
    indicators = status_indicators(
        {
            "adsb_ok": False,
            "adsb_error": (
                "Server error '502 Bad Gateway' for url "
                "'https://api.adsb.lol/v2/lat/36.587/lon/-121.843/dist/5'"
            ),
        },
        process_running=True,
        health_ok=True,
        external=False,
        config_ok=True,
        token_found=True,
    )
    provider = indicators["ADS-B provider"]
    assert provider.state == IndicatorState.FAILED
    assert provider.text == "Temporary ADSB.lol service error — retrying automatically"
    assert "https://" not in provider.text


def test_traffic_filter_status_is_compact_and_read_only() -> None:
    indicators = status_indicators(
        {
            "traffic_filter_enabled": True,
            "filtered_airline_count": 3,
            "allowed_override_count": 1,
            "unknown_operator_count": 2,
            "recent_intent_result": "likely Monterey Jet Center",
            "recent_intent_decision": "pending ADS-B confirmation",
        },
        process_running=True,
        health_ok=True,
        external=False,
        config_ok=True,
        token_found=True,
    )
    traffic = indicators["Traffic filter"]
    assert traffic.state == IndicatorState.HEALTHY
    assert "Filtered airline aircraft: 3" in traffic.text
    assert "Allowed JSX aircraft: 1" in traffic.text
    assert "likely Monterey Jet Center" in traffic.text
    assert "pending ADS-B confirmation" in traffic.text


def test_pairing_token_and_log_redaction() -> None:
    value = "GET /ws?token=supersecret HTTP/1.1 X-Pairing-Token: anothersecret"
    redacted = redact_secrets(value)
    assert "supersecret" not in redacted
    assert "anothersecret" not in redacted
    assert redacted.count("<redacted>") == 2


def test_config_validation_error(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("speech:\n  beam_size: 0\n", encoding="utf-8")
    value, error = validate_config(config)
    assert value is None
    assert error and "beam_size" in error


def test_pairing_token_path_is_relative_to_config(tmp_path: Path) -> None:
    config = AppConfig()
    assert pairing_token_path(config, tmp_path / "config.yaml") == (
        tmp_path / "data/pairing_token.txt"
    )


class FakeResponse:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.value).encode()


def test_health_polling() -> None:
    with patch("mry_alert.control.core.urlopen", return_value=FakeResponse({"status": "ok"})):
        assert fetch_json("http://127.0.0.1:8765/health") == {"status": "ok"}


def test_test_alert_request_is_authenticated_without_logging_token() -> None:
    def fake_open(request: Any, timeout: float) -> FakeResponse:
        del timeout
        assert request.get_header("X-pairing-token") == "secret"
        return FakeResponse({"test": True})

    with patch("mry_alert.control.core.urlopen", side_effect=fake_open):
        assert post_test_alert("http://127.0.0.1:8765", "secret")["test"] is True


def test_process_crash_state_allows_restart() -> None:
    ownership = ProcessOwnership(child_running=True)
    ownership.child_stopped()
    assert ownership.can_start


def test_manual_process_is_separate_from_backend_command(tmp_path: Path) -> None:
    backend = backend_command(paths(tmp_path), tmp_path / "config.yaml")
    command = manual_command(paths(tmp_path), ["check-speech-runtime"])
    assert backend[1] != command[1]


def test_pyinstaller_path_resolution(tmp_path: Path) -> None:
    executable = tmp_path / "MRY Alert Control.exe"
    backend = tmp_path / "MRY Alert Backend.exe"
    backend.touch()
    resolved = resolve_app_paths(executable=executable, frozen=True)
    assert resolved.application_dir == tmp_path
    assert resolved.backend_executable == backend
    assert resolved.config_path == tmp_path / "config.yaml"


def test_source_path_resolution() -> None:
    module = Path(__file__).parents[1] / "src/mry_alert/control/core.py"
    resolved = resolve_app_paths(module_file=module, frozen=False)
    assert resolved.application_dir.name == "ATC recognition software"
