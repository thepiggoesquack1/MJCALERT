from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtWidgets import QApplication, QPushButton

from mry_alert.audio_classifier.dataset import TrainingDataCollector
from mry_alert.audio_classifier.models import (
    AudioIntentResult,
    DestinationLabel,
    IntentLabel,
)
from mry_alert.config import TrainingDataConfig
from mry_alert.control import app as control_app
from mry_alert.control.core import AppPaths
from mry_alert.control.review import ReviewDialog


class MemorySettings:
    def __init__(self, *_: object) -> None:
        self.values: dict[str, Any] = {}

    def value(
        self, key: str, default: object = None, *, type: type[object] | None = None
    ) -> object:
        del type
        return self.values.get(key, default)

    def setValue(self, key: str, value: object) -> None:
        self.values[key] = value


def process_events_until(
    application: QApplication, predicate: Any, timeout_seconds: float = 5
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.mark.parametrize(
    ("line", "stream_default", "expected"),
    [
        ("INFO:     connection open", "ERROR", "INFO"),
        ('INFO:     127.0.0.1:63582 - "GET /health HTTP/1.1" 200 OK', "ERROR", "INFO"),
        ('INFO:     127.0.0.1:61952 - "WebSocket /ws" [accepted]', "ERROR", "INFO"),
        ("WARNING: retrying connection", "INFO", "WARNING"),
        ("ERROR: connection failed", "INFO", "ERROR"),
        ("INFO: ADS-B CORRELATION matched N12345", "ERROR", "ADS-B"),
    ],
)
def test_backend_log_level_uses_message_prefix(
    line: str, stream_default: str, expected: str
) -> None:
    assert control_app.ControlWindow._classify_line(line, stream_default) == expected


def test_copy_pairing_token_button_copies_without_displaying_or_logging_secret(
    control_window: tuple[QApplication, control_app.ControlWindow],
) -> None:
    application, window = control_window
    assert window.config is not None
    token_path = control_app.pairing_token_path(window.config, window.config_path)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    secret = "test-secret-token"
    token_path.write_text(secret + "\n", encoding="utf-8")

    window.copy_token_button.click()
    application.processEvents()

    assert QApplication.clipboard().text() == secret
    assert "copied" in window.statusBar().currentMessage().casefold()
    assert secret not in window.statusBar().currentMessage()
    assert all(secret not in line for _, line in window.log_lines)


@pytest.fixture
def control_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[QApplication, control_app.ControlWindow]:
    application = QApplication.instance() or QApplication([])
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    paths = AppPaths(
        application_dir=tmp_path,
        config_path=config,
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
        gui_logs_dir=tmp_path / "logs/gui",
        backend_executable=None,
    )
    monkeypatch.setattr(control_app, "resolve_app_paths", lambda: paths)
    monkeypatch.setattr(control_app, "QSettings", MemorySettings)
    window = control_app.ControlWindow()
    yield application, window
    if window.command_process.state() != QProcess.ProcessState.NotRunning:
        window.command_process.kill()
        window.command_process.waitForFinished(3000)
    window.poll_timer.stop()
    window.close()
    application.processEvents()


def test_long_command_does_not_freeze_and_can_be_cancelled(
    control_window: tuple[QApplication, control_app.ControlWindow],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, window = control_window
    monkeypatch.setattr(
        control_app,
        "manual_command",
        lambda _paths, _arguments: (
            sys.executable,
            ["-c", "import time; time.sleep(30)"],
        ),
    )
    timer_fired = {"value": False}
    QTimer.singleShot(25, lambda: timer_fired.__setitem__("value", True))

    window.command_input.setText("list-audio-devices")
    window.run_manual_command()

    assert process_events_until(
        application,
        lambda: window.command_process.state() == QProcess.ProcessState.Running,
    )
    assert process_events_until(application, lambda: timer_fired["value"])
    assert not window.run_button.isEnabled()
    assert window.cancel_button.isEnabled()

    window.cancel_manual_command()

    assert process_events_until(
        application,
        lambda: window.command_process.state() == QProcess.ProcessState.NotRunning,
    )
    assert window.run_button.isEnabled()
    assert not window.cancel_button.isEnabled()


def test_clean_app_exit_without_owned_backend(
    control_window: tuple[QApplication, control_app.ControlWindow],
) -> None:
    application, window = control_window
    window.show()
    application.processEvents()

    assert window.close()
    application.processEvents()
    assert not window.isVisible()


def test_alert_sensitivity_control_saves_balanced_mode(
    control_window: tuple[QApplication, control_app.ControlWindow],
) -> None:
    application, window = control_window
    assert window.sensitivity_combo.currentText() == "Conservative"

    window.sensitivity_combo.setCurrentText("Balanced")
    window.apply_alert_sensitivity()
    application.processEvents()

    assert window.config is not None
    assert window.config.detection.alert_sensitivity == "balanced"
    assert "alert_sensitivity: balanced" in window.config_path.read_text(encoding="utf-8")
    assert "incomplete pilot audio" in window.sensitivity_description.text()
    assert "#f0a43c" in window.sensitivity_combo.styleSheet()

    window.sensitivity_combo.setCurrentText("Never Miss")
    application.processEvents()
    assert "#ef6461" in window.sensitivity_combo.styleSheet()
    assert "color:#ef6461" in window.sensitivity_description.text()


def test_training_tab_toggle_updates_config_and_requests_full_restart(
    control_window: tuple[QApplication, control_app.ControlWindow],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application, window = control_window
    restart_requested: list[bool] = []
    monkeypatch.setattr(
        window,
        "_restart_entire_application",
        lambda: restart_requested.append(True),
    )
    assert window.config is not None
    assert window.config.training_data.enabled is False
    assert window.clip_recording_button.text() == "Start Recording Clips"

    window.clip_recording_button.click()
    application.processEvents()

    assert window.config.training_data.enabled is True
    assert "training_data:" in window.config_path.read_text(encoding="utf-8")
    assert "enabled: true" in window.config_path.read_text(encoding="utf-8")
    assert restart_requested == [True]


def test_notification_tab_renders_delivery_history(
    control_window: tuple[QApplication, control_app.ControlWindow],
) -> None:
    application, window = control_window
    window.notification_records = [
        {
            "event_id": "event-1",
            "sent_at": "2026-07-27T21:30:00+00:00",
            "destination": "Monterey Jet Center",
            "registration": "N123AB",
            "spoken_callsign": "November one two three alpha bravo",
            "aircraft_type": "Cessna 525",
            "confidence": 0.94,
            "transcript_excerpt": "November one two three alpha bravo to Jet Center",
            "match_reasons": ["aircraft was on the ground near KMRY"],
            "test": False,
            "connected_clients": 1,
            "delivered_clients": 1,
            "failed_clients": 0,
        },
        {
            "event_id": "event-2",
            "sent_at": "2026-07-27T21:20:00+00:00",
            "destination": "Monterey Jet Center",
            "registration": "N123AB",
            "spoken_callsign": "Test",
            "confidence": 1.0,
            "transcript_excerpt": "Local test",
            "match_reasons": ["test"],
            "test": True,
            "connected_clients": 0,
            "delivered_clients": 0,
            "failed_clients": 0,
        },
    ]

    window.render_notification_history()
    application.processEvents()

    assert window.notification_table.rowCount() == 2
    assert window.notification_total.text() == "2"
    assert window.notification_delivered.text() == "1"
    assert window.notification_table.item(0, 1).text() == "N123AB"
    assert window.notification_table.item(0, 5).text() == "Delivered to 1"
    assert "aircraft was on the ground near KMRY" in (window.notification_details.toPlainText())

    window.notification_filter.setCurrentText("Real alerts")
    application.processEvents()
    assert window.notification_table.rowCount() == 1
    assert window.notification_table.item(0, 3).text() == "Aircraft alert"


def test_training_features_are_grouped_in_dedicated_tab(
    control_window: tuple[QApplication, control_app.ControlWindow],
) -> None:
    application, window = control_window
    application.processEvents()

    assert [window.tabs.tabText(index) for index in range(window.tabs.count())] == [
        "Dashboard",
        "Notifications",
        "Event History",
        "Training",
    ]
    assert window.tabs.widget(3) is window.training_tab
    training_labels = {button.text() for button in window.training_tab.findChildren(QPushButton)}
    assert {
        "Open Review Queue",
        "Train Classifier",
        "Evaluate Model",
        "Open Clip Folder",
        "Open Model Folder",
    } <= training_labels
    dashboard_labels = {button.text() for button in window.tabs.widget(0).findChildren(QPushButton)}
    assert "Train Classifier" not in dashboard_labels
    assert "Open Review Queue" not in dashboard_labels


def test_event_history_tab_searches_and_shows_complete_detail(
    control_window: tuple[QApplication, control_app.ControlWindow],
) -> None:
    application, window = control_window
    window.backend_session_id = "session-one"
    window.event_history_records = [
        {
            "event_id": "arrival-1",
            "transition_type": "confirmed",
            "timestamp": "2026-08-05T18:01:00+00:00",
            "registration": "N627S",
            "spoken_callsign": "November six two seven sierra",
            "operator_name": "Test Operator",
            "aircraft_type_name": "Cessna Citation CJ3",
            "aircraft_type_code": "C25B",
            "destination": "Monterey Jet Center",
            "intent": "taxi_or_route_to_destination",
            "direction_state": "inbound",
            "adsb_movement_state": "taxiing_toward_destination",
            "adsb_score": 88.0,
            "winning_margin": 17.0,
            "final_decision": "confirmed",
            "notification_status": "confirmed",
            "chrome_delivery_result": "delivered",
            "ntfy_delivery_result": "delivered",
            "decision_reasons": ["strong pilot arrival evidence"],
            "transcript_excerpt": "We would like to go to Monterey Jet Center",
        },
        {
            "event_id": "outbound-1",
            "transition_type": "outbound_filtered",
            "timestamp": "2026-08-05T18:00:00+00:00",
            "registration": "N100J",
            "aircraft_type_name": "Pilatus PC-24",
            "destination": "Monterey Jet Center",
            "final_decision": "ignored",
            "decision_reasons": ["already inside Jet Center geofence"],
            "transcript_excerpt": "at the Jet Center request taxi",
        },
    ]
    window.render_event_history()
    application.processEvents()

    assert window.event_table.rowCount() == 2
    assert window.event_table.item(0, 1).text() == "N627S"
    assert window.event_table.item(0, 2).text() == "Cessna Citation CJ3"
    assert "strong pilot arrival evidence" in window.event_details.toPlainText()

    window.event_search.setText("PC-24")
    application.processEvents()
    assert window.event_table.rowCount() == 1
    assert window.event_table.item(0, 1).text() == "N100J"


def test_event_history_refresh_preserves_selected_event(
    control_window: tuple[QApplication, control_app.ControlWindow],
) -> None:
    application, window = control_window
    window.event_history_records = [
        {
            "event_id": "newest",
            "transition_type": "confirmed",
            "timestamp": "2026-08-05T18:02:00+00:00",
            "registration": "N200J",
            "final_decision": "confirmed",
        },
        {
            "event_id": "selected",
            "transition_type": "outbound_filtered",
            "timestamp": "2026-08-05T18:01:00+00:00",
            "registration": "N100J",
            "final_decision": "ignored",
        },
    ]
    window.render_event_history()
    window.event_table.selectRow(1)
    application.processEvents()

    window.event_history_records.insert(
        0,
        {
            "event_id": "new-arrival",
            "transition_type": "confirmed",
            "timestamp": "2026-08-05T18:03:00+00:00",
            "registration": "N300J",
            "final_decision": "confirmed",
        },
    )
    window.render_event_history()
    application.processEvents()

    selected = window.event_table.item(window.event_table.currentRow(), 0)
    record = selected.data(Qt.ItemDataRole.UserRole)
    assert record["event_id"] == "selected"
    assert window.event_table.currentRow() == 2


def test_training_tab_reports_pending_review_clips(
    control_window: tuple[QApplication, control_app.ControlWindow],
) -> None:
    application, window = control_window
    assert window.config is not None
    window.config.training_data.enabled = True
    dataset = window.config_path.parent / window.config.training_data.directory
    collector = TrainingDataCollector(TrainingDataConfig(enabled=True, directory=dataset))
    assert collector.save(
        b"\0\0" * 100,
        16000,
        transcript="review me",
        normalized_transcript="review me",
        classification=None,
        classifier_status="disabled",
    )

    window.update_training_tab()
    application.processEvents()

    assert window.training_collection_value.text() == "Enabled"
    assert window.training_pending_value.text() == "1"
    assert "Reviewed: 0" in window.training_details.text()


def test_gui_review_workflow_saves_human_label(
    control_window: tuple[QApplication, control_app.ControlWindow],
    tmp_path: Path,
) -> None:
    application, window = control_window
    dataset = tmp_path / "training"
    collector = TrainingDataCollector(TrainingDataConfig(enabled=True, directory=dataset))
    saved = collector.save(
        b"\0\0" * 100,
        16000,
        transcript="uncertain audio",
        normalized_transcript="uncertain audio",
        classification=AudioIntentResult(
            destination=DestinationLabel.MONTEREY_JET_CENTER,
            destination_confidence=0.55,
            intent=IntentLabel.WEAK_DESTINATION_MENTION,
            intent_confidence=0.55,
        ),
    )
    assert saved is not None
    dialog = ReviewDialog(dataset, window)
    dialog.clips.setCurrentRow(0)
    application.processEvents()
    dialog.destination.setCurrentText(DestinationLabel.NONE)
    dialog.intent.setCurrentText(IntentLabel.NONE)
    dialog.save_review()
    assert len(list((dataset / "reviewed").glob("*_metadata.json"))) == 1
    dialog.close()


def test_gui_review_queue_skips_invalid_clip_with_explanation(
    control_window: tuple[QApplication, control_app.ControlWindow],
    tmp_path: Path,
) -> None:
    application, window = control_window
    dataset = tmp_path / "training"
    pending = dataset / "pending"
    pending.mkdir(parents=True)
    (pending / "broken_metadata.json").write_text("{", encoding="utf-8")

    dialog = ReviewDialog(dataset, window)
    application.processEvents()

    assert dialog.clips.count() == 0
    assert "Skipped 1 invalid item" in dialog.summary.toPlainText()
    assert "corrupted_metadata" in dialog.summary.toPlainText()
    dialog.close()
