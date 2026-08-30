from __future__ import annotations

import json
import os
import re
import shlex
import sys
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from mry_alert.config import AppConfig

EVENT_HISTORY_FILTERS = {
    "all": None,
    "confirmed": {"confirmed"},
    "possible": {"possible"},
    "pending": {"pending"},
    "expired": {"expired"},
    "corrected": {"corrected"},
    "cancelled": {"cancelled"},
    "denied": {"denied", "suppressed"},
    "unresolved": {"unresolved"},
    "ambiguous": {"ambiguous"},
    "outbound filtered": {"outbound_filtered"},
    "airline filtered": {"airline_filtered"},
    "duplicate": {"duplicate"},
}


def filter_session_events(
    records: list[dict[str, Any]],
    query: str = "",
    category: str = "all",
    sort_order: str = "newest first",
) -> list[dict[str, Any]]:
    """Search, filter, and sort backend-owned session records for display."""
    needle = query.casefold().strip()
    allowed = EVENT_HISTORY_FILTERS.get(category.casefold())
    searchable = (
        "registration",
        "spoken_callsign",
        "operator_name",
        "aircraft_type",
        "aircraft_type_code",
        "aircraft_type_name",
        "manufacturer",
        "model",
        "destination",
        "transcript_excerpt",
        "final_decision",
        "event_id",
        "operator_acknowledgement",
    )

    def matches(record: dict[str, Any]) -> bool:
        transition = str(record.get("transition_type", "")).casefold()
        if category.casefold() == "delivery failed":
            if not any(
                str(record.get(field, "")).casefold() in {"failed", "partial"}
                or "fail" in str(record.get(field, "")).casefold()
                for field in (
                    "notification_status",
                    "chrome_delivery_result",
                    "ntfy_delivery_result",
                )
            ):
                return False
        elif allowed is not None and transition not in allowed:
            return False
        if not needle:
            return True
        values = [str(record.get(field) or "") for field in searchable]
        values.extend(str(reason) for reason in record.get("decision_reasons", []))
        return needle in " ".join(values).casefold()

    result = [record for record in records if matches(record)]
    order = sort_order.casefold()
    if order == "oldest first":
        return sorted(result, key=lambda item: str(item.get("timestamp", "")))
    if order == "tail number":
        return sorted(result, key=lambda item: str(item.get("registration") or "~").casefold())
    if order == "aircraft type":
        return sorted(
            result,
            key=lambda item: str(
                item.get("aircraft_type_name") or item.get("aircraft_type") or "~"
            ).casefold(),
        )
    if order == "decision":
        return sorted(result, key=lambda item: str(item.get("final_decision", "")).casefold())
    return sorted(result, key=lambda item: str(item.get("timestamp", "")), reverse=True)

SUPPORTED_COMMANDS = {
    "check-speech-runtime",
    "compare-models",
    "replay",
    "evaluate-models",
    "list-audio-devices",
    "monitor-microphone",
    "transcribe-file",
    "simulate",
    "train-audio-classifier",
    "evaluate-audio-classifier",
}
DANGEROUS_SHELL = re.compile(r"&&|\|\||>>|\$\(|[&|><;`]")
TOKEN_PATTERN = re.compile(r"(?i)(token=)[^&\s\"']+")
HEADER_TOKEN_PATTERN = re.compile(r"(?i)(x-pairing-token\s*[:=]\s*)\S+")


class IndicatorState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Indicator:
    state: IndicatorState
    text: str


@dataclass(frozen=True)
class AppPaths:
    application_dir: Path
    config_path: Path
    data_dir: Path
    logs_dir: Path
    gui_logs_dir: Path
    backend_executable: Path | None


@dataclass
class ProcessOwnership:
    child_running: bool = False
    external_detected: bool = False

    @property
    def can_start(self) -> bool:
        return not self.child_running and not self.external_detected

    @property
    def can_stop(self) -> bool:
        return self.child_running

    def child_started(self) -> None:
        if self.external_detected:
            raise RuntimeError("An external backend is already running.")
        self.child_running = True

    def child_stopped(self) -> None:
        self.child_running = False

    def observe_health(self, healthy: bool) -> None:
        self.external_detected = healthy and not self.child_running


def resolve_app_paths(
    *,
    module_file: Path | None = None,
    executable: Path | None = None,
    frozen: bool | None = None,
) -> AppPaths:
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    exe = executable or Path(sys.executable)
    if is_frozen:
        application_dir = exe.resolve().parent
        bundled_backend = application_dir / "MRY Alert Backend.exe"
        backend = bundled_backend if bundled_backend.exists() else None
    else:
        source = (module_file or Path(__file__)).resolve()
        application_dir = source.parents[3]
        backend = None
    logs = application_dir / "logs"
    return AppPaths(
        application_dir=application_dir,
        config_path=application_dir / "config.yaml",
        data_dir=application_dir / "data",
        logs_dir=logs,
        gui_logs_dir=logs / "gui",
        backend_executable=backend,
    )


def _strip_windows_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def parse_safe_command(command: str) -> list[str]:
    command = command.strip()
    if not command:
        raise ValueError("Enter an mry_alert command.")
    if DANGEROUS_SHELL.search(command):
        raise ValueError("Shell operators are not allowed.")
    try:
        parts = [_strip_windows_quotes(part) for part in shlex.split(command, posix=False)]
    except ValueError as exc:
        raise ValueError(f"Invalid command quoting: {exc}") from exc
    if not parts or parts[0] not in SUPPORTED_COMMANDS:
        supported = ", ".join(sorted(SUPPORTED_COMMANDS))
        raise ValueError(f"Unsupported command. Allowed commands: {supported}")
    return parts


def backend_command(paths: AppPaths, config_path: Path) -> tuple[str, list[str]]:
    if paths.backend_executable:
        return str(paths.backend_executable), ["serve", "--config", str(config_path)]
    return sys.executable, ["-m", "mry_alert", "serve", "--config", str(config_path)]


def manual_command(paths: AppPaths, arguments: list[str]) -> tuple[str, list[str]]:
    if paths.backend_executable:
        return str(paths.backend_executable), arguments
    return sys.executable, ["-m", "mry_alert", *arguments]


def display_command(program: str, arguments: list[str]) -> str:
    return subprocess_list2cmdline([program, *arguments])


def subprocess_list2cmdline(arguments: list[str]) -> str:
    import subprocess

    return subprocess.list2cmdline(arguments)


def redact_secrets(text: str) -> str:
    text = TOKEN_PATTERN.sub(r"\1<redacted>", text)
    return HEADER_TOKEN_PATTERN.sub(r"\1<redacted>", text)


def validate_config(path: Path) -> tuple[AppConfig | None, str | None]:
    try:
        return AppConfig.load(path), None
    except (OSError, ValueError) as exc:
        return None, str(exc)


def update_alert_sensitivity(path: Path, mode: str) -> None:
    """Atomically update only detection.alert_sensitivity in an existing YAML file."""
    allowed = {"conservative", "balanced", "never_miss"}
    if mode not in allowed:
        raise ValueError(f"Unsupported alert sensitivity: {mode}")
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    if original.strip() == "{}":
        lines = []
    detection_index = next(
        (index for index, line in enumerate(lines) if line.strip() == "detection:"),
        None,
    )
    newline = "\r\n" if "\r\n" in original else "\n"
    setting = f"  alert_sensitivity: {mode}{newline}"
    if detection_index is None:
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += newline
        lines.extend([f"detection:{newline}", setting])
    else:
        end = len(lines)
        for index in range(detection_index + 1, len(lines)):
            line = lines[index]
            if line.strip() and not line.startswith((" ", "\t", "#")):
                end = index
                break
        existing = next(
            (
                index
                for index in range(detection_index + 1, end)
                if lines[index].lstrip().startswith("alert_sensitivity:")
            ),
            None,
        )
        if existing is None:
            lines.insert(detection_index + 1, setting)
        else:
            lines[existing] = setting
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text("".join(lines), encoding="utf-8", newline="")
        AppConfig.load(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def update_training_collection(path: Path, enabled: bool) -> None:
    """Atomically update only training_data.enabled in an existing YAML file."""
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    if original.strip() == "{}":
        lines = []
    section_index = next(
        (index for index, line in enumerate(lines) if line.strip() == "training_data:"),
        None,
    )
    newline = "\r\n" if "\r\n" in original else "\n"
    setting = f"  enabled: {'true' if enabled else 'false'}{newline}"
    if section_index is None:
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += newline
        lines.extend([f"training_data:{newline}", setting])
    else:
        end = len(lines)
        for index in range(section_index + 1, len(lines)):
            line = lines[index]
            if line.strip() and not line.startswith((" ", "\t", "#")):
                end = index
                break
        existing = next(
            (
                index
                for index in range(section_index + 1, end)
                if lines[index].lstrip().startswith("enabled:")
            ),
            None,
        )
        if existing is None:
            lines.insert(section_index + 1, setting)
        else:
            indentation = lines[existing][: len(lines[existing]) - len(lines[existing].lstrip())]
            comment = ""
            if "#" in lines[existing]:
                comment = " #" + lines[existing].split("#", 1)[1].rstrip("\r\n")
            lines[existing] = (
                f"{indentation}enabled: {'true' if enabled else 'false'}{comment}{newline}"
            )
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text("".join(lines), encoding="utf-8", newline="")
        AppConfig.load(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def pairing_token_path(config: AppConfig, config_path: Path) -> Path:
    configured = config.server.pairing_token_file
    return configured if configured.is_absolute() else config_path.parent / configured


def read_pairing_token(config: AppConfig, config_path: Path) -> str:
    path = pairing_token_path(config, config_path)
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"Pairing token file is empty: {path}")
    return token


def fetch_json(url: str, timeout: float = 1.5) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    if not isinstance(value, dict):
        raise RuntimeError("Backend returned a non-object response.")
    return value


def post_test_alert(base_url: str, token: str, timeout: float = 3) -> dict[str, Any]:
    request = Request(
        f"{base_url.rstrip('/')}/api/test-alert",
        method="POST",
        headers={"X-Pairing-Token": token},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc
    if not isinstance(value, dict):
        raise RuntimeError("Backend returned a non-object response.")
    return value


def backend_indicator(process_running: bool, health_ok: bool, external: bool = False) -> Indicator:
    if health_ok and external:
        return Indicator(IndicatorState.HEALTHY, "External backend detected")
    if health_ok:
        return Indicator(IndicatorState.HEALTHY, "Running")
    if process_running:
        return Indicator(IndicatorState.DEGRADED, "Starting or health unavailable")
    return Indicator(IndicatorState.FAILED, "Stopped")


def status_indicators(
    status: dict[str, Any] | None,
    *,
    process_running: bool,
    health_ok: bool,
    external: bool,
    config_ok: bool,
    token_found: bool,
) -> dict[str, Indicator]:
    result = {
        "Backend server": backend_indicator(process_running, health_ok, external),
        "Health endpoint": Indicator(
            IndicatorState.HEALTHY if health_ok else IndicatorState.FAILED,
            "Responding" if health_ok else "Unavailable",
        ),
        "Config file": Indicator(
            IndicatorState.HEALTHY if config_ok else IndicatorState.FAILED,
            "Valid" if config_ok else "Invalid or missing",
        ),
        "Pairing token": Indicator(
            IndicatorState.HEALTHY if token_found else IndicatorState.FAILED,
            "Found" if token_found else "Missing",
        ),
    }
    if not status:
        unknown = Indicator(IndicatorState.UNKNOWN, "Unknown")
        for name in (
            "Chrome extension",
            "LiveATC audio",
            "Audio stream activity",
            "Speech model",
            "ADS-B provider",
            "ADS-B aircraft tracking",
            "Traffic filter",
            "Notification delivery",
            "ntfy push",
            "Last transcription",
            "Last ADS-B correlation",
            "Audio classifier",
            "Dataset collection",
        ):
            result[name] = unknown
        return result
    extension_count = int(status.get("extension_clients", status.get("connected_extensions", 0)))
    result["Chrome extension"] = Indicator(
        IndicatorState.HEALTHY if extension_count else IndicatorState.DEGRADED,
        f"Connected — {extension_count} client(s)" if extension_count else "No client connected",
    )
    audio_connected = bool(status.get("audio_connected"))
    result["LiveATC audio"] = Indicator(
        IndicatorState.HEALTHY if audio_connected else IndicatorState.DEGRADED,
        "Connected" if audio_connected else "Waiting for audio connection",
    )
    audio_recent = bool(status.get("audio_recently_received"))
    result["Audio stream activity"] = Indicator(
        IndicatorState.HEALTHY if audio_recent else IndicatorState.DEGRADED,
        "Audio received recently"
        if audio_recent
        else ("Connected, but inactive for 15 seconds" if audio_connected else "No audio"),
    )
    speech_ready = bool(status.get("speech_ready"))
    model = str(status.get("speech_model", "unknown"))
    result["Speech model"] = Indicator(
        IndicatorState.HEALTHY if speech_ready else IndicatorState.DEGRADED,
        f"{model} ready" if speech_ready else f"{model} not loaded",
    )
    adsb_ok = bool(status.get("adsb_ok"))
    adsb_error = status.get("adsb_error")
    adsb_error_text = str(adsb_error)
    if adsb_error and any(marker in adsb_error_text for marker in ("502", "503", "504")):
        adsb_error_text = "Temporary ADSB.lol service error — retrying automatically"
    elif adsb_error and "timed out" in adsb_error_text.lower():
        adsb_error_text = "ADSB.lol timed out — retrying automatically"
    tracked = int(status.get("adsb_tracked_aircraft", 0))
    adsb_state = (
        IndicatorState.FAILED
        if adsb_error
        else IndicatorState.HEALTHY
        if adsb_ok and tracked
        else IndicatorState.DEGRADED
    )
    result["ADS-B provider"] = Indicator(
        adsb_state,
        adsb_error_text
        if adsb_error
        else f"{status.get('adsb_provider', 'unknown')} — {tracked} tracked",
    )
    result["ADS-B aircraft tracking"] = Indicator(
        IndicatorState.HEALTHY if tracked else IndicatorState.DEGRADED,
        f"{tracked} aircraft tracked" if tracked else "No aircraft tracked",
    )
    traffic_filter_enabled = bool(status.get("traffic_filter_enabled"))
    result["Traffic filter"] = Indicator(
        IndicatorState.HEALTHY if traffic_filter_enabled else IndicatorState.UNKNOWN,
        (
            "Enabled\n"
            f"Filtered airline aircraft: {int(status.get('filtered_airline_count', 0))}\n"
            f"Allowed JSX aircraft: {int(status.get('allowed_override_count', 0))}\n"
            f"Unknown operators allowed: {int(status.get('unknown_operator_count', 0))}\n"
            f"Recent intent: {status.get('recent_intent_result', 'none')}\n"
            f"Recent decision: {status.get('recent_intent_decision', 'none')}"
        )
        if traffic_filter_enabled
        else "Disabled",
    )
    delivered = int(status.get("last_notification_delivered", 0))
    notification_success = status.get("last_notification_success")
    result["Notification delivery"] = Indicator(
        IndicatorState.HEALTHY
        if notification_success
        else IndicatorState.FAILED
        if notification_success is False
        else IndicatorState.UNKNOWN,
        f"Sent to {delivered} client(s)"
        if notification_success
        else "Failed"
        if notification_success is False
        else "No result yet",
    )
    ntfy_enabled = bool(status.get("ntfy_enabled"))
    ntfy_success = status.get("ntfy_last_success")
    ntfy_topic = str(status.get("ntfy_topic") or "not configured")
    ntfy_error = status.get("ntfy_last_error")
    result["ntfy push"] = Indicator(
        IndicatorState.FAILED
        if ntfy_enabled and ntfy_success is False
        else IndicatorState.HEALTHY
        if ntfy_enabled
        else IndicatorState.UNKNOWN,
        (
            f"Enabled — topic: {ntfy_topic}\n"
            + (
                f"Last delivery failed: {ntfy_error}"
                if ntfy_success is False
                else "Last delivery succeeded"
                if ntfy_success
                else "Waiting for first alert"
            )
        )
        if ntfy_enabled
        else "Disabled",
    )
    result["Last transcription"] = Indicator(
        IndicatorState.HEALTHY if status.get("last_transcription_at") else IndicatorState.UNKNOWN,
        str(status.get("last_transcription_at") or "None yet"),
    )
    result["Last ADS-B correlation"] = Indicator(
        IndicatorState.HEALTHY
        if status.get("last_adsb_correlation_at")
        else IndicatorState.UNKNOWN,
        str(status.get("last_adsb_correlation_at") or "None yet"),
    )
    classifier_enabled = bool(status.get("classifier_enabled"))
    classifier_loaded = bool(status.get("classifier_loaded"))
    classifier_error = status.get("classifier_error")
    result["Audio classifier"] = Indicator(
        IndicatorState.FAILED
        if classifier_error
        else IndicatorState.HEALTHY
        if classifier_enabled and classifier_loaded
        else IndicatorState.UNKNOWN
        if not classifier_enabled
        else IndicatorState.DEGRADED,
        str(classifier_error)
        if classifier_error
        else (
            f"{status.get('classifier_model_version') or 'model'} loaded"
            if classifier_loaded
            else "Disabled"
            if not classifier_enabled
            else "Not loaded"
        ),
    )
    collecting = bool(status.get("dataset_collection_enabled"))
    pending_reviews = int(status.get("pending_review_clips", 0))
    result["Dataset collection"] = Indicator(
        IndicatorState.DEGRADED if collecting else IndicatorState.UNKNOWN,
        f"Enabled — {pending_reviews} pending review"
        if collecting
        else "Disabled (no audio saved)",
    )
    return result


def set_windows_startup(
    enabled: bool, executable: Path, arguments: list[str] | None = None
) -> None:
    if os.name != "nt":
        raise RuntimeError("Windows startup registration is only available on Windows.")
    import winreg

    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
        if enabled:
            winreg.SetValueEx(
                key,
                "MRY Alert Control",
                0,
                winreg.REG_SZ,
                subprocess_list2cmdline([str(executable), *(arguments or [])]),
            )
        else:
            with suppress(FileNotFoundError):
                winreg.DeleteValue(key, "MRY Alert Control")
