from __future__ import annotations

import html
import json
import os
import sys
from collections import deque
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    QByteArray,
    QEvent,
    QObject,
    QProcess,
    QProcessEnvironment,
    QSettings,
    Qt,
    QTimer,
    QUrl,
)
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QDesktopServices,
    QIcon,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
)
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QSystemTrayIcon,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mry_alert.audio_classifier.dataset import scan_clips
from mry_alert.control.core import (
    Indicator,
    IndicatorState,
    backend_command,
    display_command,
    filter_session_events,
    manual_command,
    pairing_token_path,
    parse_safe_command,
    read_pairing_token,
    redact_secrets,
    resolve_app_paths,
    set_windows_startup,
    status_indicators,
    update_alert_sensitivity,
    update_training_collection,
    validate_config,
)
from mry_alert.control.review import ReviewDialog

COLORS = {
    IndicatorState.HEALTHY: "#39c879",
    IndicatorState.DEGRADED: "#f0b44d",
    IndicatorState.FAILED: "#ef6461",
    IndicatorState.UNKNOWN: "#7e8794",
}
SYMBOLS = {
    IndicatorState.HEALTHY: "✓",
    IndicatorState.DEGRADED: "⚠",
    IndicatorState.FAILED: "✕",
    IndicatorState.UNKNOWN: "●",
}


class HistoryFilter(QObject):
    def __init__(self, owner: ControlWindow) -> None:
        super().__init__(owner)
        self.owner = owner

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()  # type: ignore[attr-defined]
            if key == Qt.Key.Key_Up:
                self.owner.navigate_history(-1)
                return True
            if key == Qt.Key.Key_Down:
                self.owner.navigate_history(1)
                return True
        return super().eventFilter(watched, event)


class StatusCard(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("statusCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(3)
        self.title = QLabel(title)
        self.title.setObjectName("cardTitle")
        self.value = QLabel("● Unknown")
        self.value.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.value)

    def update_indicator(self, indicator: Indicator) -> None:
        color = COLORS[indicator.state]
        self.value.setText(f"{SYMBOLS[indicator.state]} {indicator.text}")
        self.value.setStyleSheet(f"color: {color}; font-weight: 600;")


class ControlWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.paths = resolve_app_paths()
        self.config_path = self.paths.config_path
        self.config, self.config_error = validate_config(self.config_path)
        self.backend_process = QProcess(self)
        self.command_process = QProcess(self)
        self.backend_owned = False
        self.external_backend = False
        self.health_ok = False
        self.status_payload: dict[str, Any] | None = None
        self.notification_records: list[dict[str, Any]] = []
        self.event_history_records: list[dict[str, Any]] = []
        self.backend_session_id: str | None = None
        self.command_history: list[str] = []
        self.history_index = 0
        self.log_lines: deque[tuple[str, str]] = deque(maxlen=5000)
        self.display_paused = False
        self._full_relaunch_in_progress = False
        self._start_backend_after_relaunch = (
            os.environ.pop("MRY_ALERT_RESTART_BACKEND", "") == "1"
        )
        self.settings = QSettings("MRY Jet Center", "MRY Alert Control")
        self.network = QNetworkAccessManager(self)
        self._health_reply: QNetworkReply | None = None
        self._status_reply: QNetworkReply | None = None
        self._notifications_reply: QNetworkReply | None = None
        self._history_reply: QNetworkReply | None = None
        self._acknowledgement_reply: QNetworkReply | None = None
        self._build_ui()
        self._wire_processes()
        self._create_tray()
        self._restore_geometry()
        self.paths.gui_logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_log = self.paths.gui_logs_dir / f"control-{datetime.now():%Y%m%d-%H%M%S}.log"
        self.append_log("INFO", f"Control application started in {self.paths.application_dir}")
        self.poll_backend()
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_backend)
        self.poll_timer.start(2000)
        if self._start_backend_after_relaunch:
            QTimer.singleShot(500, self.start_server)

    @property
    def base_url(self) -> str:
        if self.config:
            return f"http://{self.config.server.host}:{self.config.server.port}"
        return "http://127.0.0.1:8765"

    def _build_ui(self) -> None:
        self.setWindowTitle("MRY Alert Control")
        self.resize(1240, 850)
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        dashboard = QWidget()
        root = QVBoxLayout(dashboard)
        root.setContentsMargins(16, 14, 16, 12)
        root.setSpacing(12)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("MRY Alert Control")
        title.setObjectName("mainTitle")
        subtitle = QLabel("Local backend launcher and operational dashboard")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        self.config_label = QLabel(str(self.config_path))
        self.config_label.setToolTip("Active backend configuration")
        header.addWidget(self.config_label)
        root.addLayout(header)

        self.cards: dict[str, StatusCard] = {}
        card_grid = QGridLayout()
        card_grid.setSpacing(8)
        names = (
            "Backend server",
            "Health endpoint",
            "Chrome extension",
            "LiveATC audio",
            "Audio stream activity",
            "Speech model",
            "ADS-B provider",
            "ADS-B aircraft tracking",
            "Traffic filter",
            "Notification delivery",
            "ntfy push",
            "Config file",
            "Pairing token",
            "Last transcription",
            "Last ADS-B correlation",
        )
        for index, name in enumerate(names):
            card = StatusCard(name)
            self.cards[name] = card
            card_grid.addWidget(card, index // 4, index % 4)
        root.addLayout(card_grid)

        controls = QHBoxLayout()
        self.start_button = self._button("Start Server", self.start_server)
        self.stop_button = self._button("Stop Server", self.stop_server)
        self.restart_button = self._button("Restart Server", self.restart_server)
        self.test_button = self._button("Test Alert", self.test_alert)
        self.open_config_button = self._button("Open Config", self.open_config)
        self.select_config_button = self._button("Select Config", self.select_config)
        self.open_logs_button = self._button("Open Logs", self.open_logs)
        self.copy_token_button = self._button("Copy Pairing Token", self.copy_pairing_token)
        token_button = self._button("Token Folder", self.open_token_folder)
        clear_files_button = self._button("Clear Logs", self.clear_log_files)
        exit_button = self._button("Exit", self.close)
        for button in (
            self.start_button,
            self.stop_button,
            self.restart_button,
            self.test_button,
            self.open_config_button,
            self.select_config_button,
            self.open_logs_button,
            self.copy_token_button,
            token_button,
            clear_files_button,
            exit_button,
        ):
            controls.addWidget(button)
        controls.addStretch()
        root.addLayout(controls)

        sensitivity_panel = QFrame()
        sensitivity_panel.setObjectName("activityPanel")
        sensitivity_layout = QHBoxLayout(sensitivity_panel)
        sensitivity_text = QVBoxLayout()
        sensitivity_title = QLabel("Alert sensitivity")
        sensitivity_title.setObjectName("cardTitle")
        self.sensitivity_description = QLabel()
        self.sensitivity_description.setWordWrap(True)
        self.sensitivity_description.setObjectName("subtitle")
        sensitivity_text.addWidget(sensitivity_title)
        sensitivity_text.addWidget(self.sensitivity_description)
        sensitivity_layout.addLayout(sensitivity_text, 1)
        self.sensitivity_combo = QComboBox()
        self.sensitivity_combo.addItems(["Conservative", "Balanced", "Never Miss"])
        self.sensitivity_combo.setItemData(0, QColor("#39c879"), Qt.ItemDataRole.ForegroundRole)
        self.sensitivity_combo.setItemData(1, QColor("#f0a43c"), Qt.ItemDataRole.ForegroundRole)
        self.sensitivity_combo.setItemData(2, QColor("#ef6461"), Qt.ItemDataRole.ForegroundRole)
        self.sensitivity_combo.currentTextChanged.connect(
            self._update_sensitivity_description
        )
        sensitivity_layout.addWidget(self.sensitivity_combo)
        sensitivity_layout.addWidget(
            self._button("Apply & Restart", self.apply_alert_sensitivity)
        )
        root.addWidget(sensitivity_panel)
        self._sync_alert_sensitivity()

        splitter = QSplitter(Qt.Orientation.Vertical)
        upper = QWidget()
        upper_layout = QVBoxLayout(upper)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        upper_layout.addWidget(QLabel("Command Console"))
        command_row = QHBoxLayout()
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText(
            'Example: check-speech-runtime --config "config.yaml"'
        )
        self.command_input.returnPressed.connect(self.run_manual_command)
        self.history_filter = HistoryFilter(self)
        self.command_input.installEventFilter(self.history_filter)
        self.run_button = self._button("Run", self.run_manual_command)
        self.cancel_button = self._button("Cancel Command", self.cancel_manual_command)
        clear_command = self._button("Clear", self.command_input.clear)
        copy_output = self._button("Copy Output", self.copy_logs)
        command_row.addWidget(self.command_input, 1)
        command_row.addWidget(self.run_button)
        command_row.addWidget(self.cancel_button)
        command_row.addWidget(clear_command)
        command_row.addWidget(copy_output)
        upper_layout.addLayout(command_row)

        activity = QFrame()
        activity.setObjectName("activityPanel")
        activity_layout = QVBoxLayout(activity)
        activity_layout.addWidget(QLabel("Recent Detection"))
        self.activity_label = QLabel("No qualifying event yet")
        self.activity_label.setWordWrap(True)
        self.activity_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        activity_layout.addWidget(self.activity_label)
        upper_layout.addWidget(activity)
        splitter.addWidget(upper)

        lower = QWidget()
        lower_layout = QVBoxLayout(lower)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("Live Backend and Command Logs"))
        self.log_filter = QComboBox()
        self.log_filter.addItems(
            [
                "ALL",
                "INFO",
                "WARNING",
                "ERROR",
                "TRANSCRIPTION",
                "ADS-B",
                "NOTIFICATION",
            ]
        )
        self.log_filter.currentTextChanged.connect(self.rebuild_logs)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search logs")
        self.search_input.textChanged.connect(self.rebuild_logs)
        self.autoscroll = QCheckBox("Auto-scroll")
        self.autoscroll.setChecked(True)
        self.pause_display = QCheckBox("Pause display")
        self.pause_display.toggled.connect(self._set_pause)
        clear_logs = self._button("Clear Display", self.clear_log_display)
        save_logs = self._button("Save Logs", self.save_logs)
        log_header.addStretch()
        log_header.addWidget(self.log_filter)
        log_header.addWidget(self.search_input)
        log_header.addWidget(self.autoscroll)
        log_header.addWidget(self.pause_display)
        log_header.addWidget(clear_logs)
        log_header.addWidget(save_logs)
        lower_layout.addLayout(log_header)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setAcceptRichText(True)
        self.log_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        lower_layout.addWidget(self.log_view)
        splitter.addWidget(lower)
        splitter.setSizes([260, 440])
        root.addWidget(splitter, 1)

        footer = QHBoxLayout()
        self.startup_checkbox = QCheckBox("Start MRY Alert Control when Windows starts")
        self.startup_checkbox.toggled.connect(self.toggle_startup)
        self.minimize_checkbox = QCheckBox("Minimize to tray")
        self.minimize_checkbox.setChecked(
            bool(self.settings.value("minimize_to_tray", True, type=bool))
        )
        footer.addWidget(self.startup_checkbox)
        footer.addWidget(self.minimize_checkbox)
        footer.addStretch()
        root.addLayout(footer)
        self.tabs.addTab(dashboard, "Dashboard")
        self.tabs.addTab(self._build_notifications_tab(), "Notifications")
        self.tabs.addTab(self._build_event_history_tab(), "Event History")
        self.training_tab = self._build_training_tab()
        self.tabs.addTab(self.training_tab, "Training")
        central_layout.addWidget(self.tabs)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")
        self.setStyleSheet(self._stylesheet())
        self._shortcuts()
        self._refresh_buttons()

    def _build_training_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("trainingTab")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        heading_row = QHBoxLayout()
        heading_box = QVBoxLayout()
        heading = QLabel("Training and review")
        heading.setObjectName("sectionTitle")
        description = QLabel(
            "Review automatically collected radio clips, train the local classifier, "
            "and evaluate it offline."
        )
        description.setObjectName("subtitle")
        heading_box.addWidget(heading)
        heading_box.addWidget(description)
        heading_row.addLayout(heading_box)
        heading_row.addStretch()
        heading_row.addWidget(self._button("Refresh", self.refresh_training_status))
        layout.addLayout(heading_row)

        metrics = QHBoxLayout()
        collection_card, self.training_collection_value = self._notification_metric(
            "Automatic collection", "Checking"
        )
        pending_card, self.training_pending_value = self._notification_metric(
            "Waiting for review", "0"
        )
        classifier_card, self.training_classifier_value = self._notification_metric(
            "Audio classifier", "Checking"
        )
        for card in (collection_card, pending_card, classifier_card):
            metrics.addWidget(card)
        layout.addLayout(metrics)

        workflow = QFrame()
        workflow.setObjectName("activityPanel")
        workflow_layout = QVBoxLayout(workflow)
        workflow_title = QLabel("Review workflow")
        workflow_title.setObjectName("sectionTitle")
        workflow_text = QLabel(
            "1. Monitoring automatically creates one clip per transmission.\n"
            "2. Open the review queue, select a clip, and use Play Clip.\n"
            "3. Correct the destination, intent, callsign, or hard-negative label.\n"
            "4. Save the review. Train uses only valid human-reviewed clips."
        )
        workflow_text.setWordWrap(True)
        workflow_layout.addWidget(workflow_title)
        workflow_layout.addWidget(workflow_text)
        layout.addWidget(workflow)

        actions = QFrame()
        actions.setObjectName("activityPanel")
        actions_layout = QVBoxLayout(actions)
        actions_title = QLabel("Actions")
        actions_title.setObjectName("sectionTitle")
        actions_layout.addWidget(actions_title)
        action_row = QHBoxLayout()
        self.review_button = self._button("Open Review Queue", self.open_review_queue)
        self.train_button = self._button("Train Classifier", self.train_classifier)
        self.evaluate_button = self._button("Evaluate Model", self.evaluate_classifier)
        self.clip_recording_button = self._button(
            "Start Recording Clips", self.toggle_clip_recording
        )
        action_row.addWidget(self.clip_recording_button)
        action_row.addWidget(self.review_button)
        action_row.addWidget(self.train_button)
        action_row.addWidget(self.evaluate_button)
        action_row.addWidget(self._button("Open Clip Folder", self.open_training_folder))
        action_row.addWidget(self._button("Open Model Folder", self.open_model_folder))
        action_row.addWidget(self._button("Clear Output", self.clear_training_output))
        action_row.addStretch()
        actions_layout.addLayout(action_row)
        self.training_details = QLabel()
        self.training_details.setObjectName("subtitle")
        self.training_details.setWordWrap(True)
        self.training_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        actions_layout.addWidget(self.training_details)
        layout.addWidget(actions)

        output_title = QLabel("Training command output")
        output_title.setObjectName("cardTitle")
        layout.addWidget(output_title)
        self.training_output = QTextEdit()
        self.training_output.setReadOnly(True)
        self.training_output.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.training_output.setPlaceholderText("Train and Evaluate output will appear here.")
        layout.addWidget(self.training_output, 1)
        self.update_training_tab()
        return tab

    def _build_notifications_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        heading_row = QHBoxLayout()
        heading_box = QVBoxLayout()
        heading = QLabel("Recent notifications")
        heading.setObjectName("sectionTitle")
        description = QLabel(
            "A delivery history of alerts sent by the backend to the Chrome extension."
        )
        description.setObjectName("subtitle")
        heading_box.addWidget(heading)
        heading_box.addWidget(description)
        heading_row.addLayout(heading_box)
        heading_row.addStretch()
        self.notification_filter = QComboBox()
        self.notification_filter.addItems(["All notifications", "Real alerts", "Test alerts"])
        self.notification_filter.currentTextChanged.connect(self.render_notification_history)
        refresh_button = self._button("Refresh", self.refresh_notifications)
        heading_row.addWidget(self.notification_filter)
        heading_row.addWidget(refresh_button)
        layout.addLayout(heading_row)

        ntfy_panel = QFrame()
        ntfy_panel.setObjectName("activityPanel")
        ntfy_layout = QVBoxLayout(ntfy_panel)
        ntfy_title = QLabel("ntfy push notifications")
        ntfy_title.setObjectName("sectionTitle")
        self.ntfy_details = QLabel(
            "Disabled. Add the ntfy section to config.yaml to enable phone push alerts."
        )
        self.ntfy_details.setWordWrap(True)
        self.ntfy_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        ntfy_layout.addWidget(ntfy_title)
        ntfy_layout.addWidget(self.ntfy_details)
        layout.addWidget(ntfy_panel)

        metrics = QHBoxLayout()
        recent_card, self.notification_total = self._notification_metric("Recent", "0")
        delivered_card, self.notification_delivered = self._notification_metric("Delivered", "0")
        latest_card, self.notification_last = self._notification_metric("Latest", "None yet")
        for card in (recent_card, delivered_card, latest_card):
            metrics.addWidget(card)
        layout.addLayout(metrics)

        self.notification_table = QTableWidget(0, 6)
        self.notification_table.setObjectName("notificationTable")
        self.notification_table.setHorizontalHeaderLabels(
            ["Time", "Aircraft", "Destination", "Type", "Confidence", "Delivery"]
        )
        header = self.notification_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.notification_table.verticalHeader().setVisible(False)
        self.notification_table.setAlternatingRowColors(True)
        self.notification_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.notification_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.notification_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.notification_table.itemSelectionChanged.connect(self._show_selected_notification)
        layout.addWidget(self.notification_table, 1)

        self.notification_details = QTextEdit()
        self.notification_details.setObjectName("notificationDetails")
        self.notification_details.setReadOnly(True)
        self.notification_details.setMaximumHeight(150)
        self.notification_details.setPlaceholderText(
            "Select a notification to see the transcript and matching reasons."
        )
        layout.addWidget(self.notification_details)
        notification_actions = QHBoxLayout()
        notification_actions.addWidget(
            self._button("Seen", lambda: self.acknowledge_selected_event("seen", False))
        )
        notification_actions.addWidget(
            self._button(
                "Aircraft Arrived",
                lambda: self.acknowledge_selected_event("aircraft_arrived", False),
            )
        )
        notification_actions.addWidget(
            self._button(
                "False Detection",
                lambda: self.acknowledge_selected_event("false_detection", False),
            )
        )
        notification_actions.addStretch()
        layout.addLayout(notification_actions)
        return tab

    def _build_event_history_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        heading = QLabel("Event History")
        heading.setObjectName("sectionTitle")
        description = QLabel(
            "Every aircraft decision from this backend session. History clears when the "
            "backend stops and is never saved automatically."
        )
        description.setObjectName("subtitle")
        layout.addWidget(heading)
        layout.addWidget(description)

        controls = QHBoxLayout()
        self.event_search = QLineEdit()
        self.event_search.setPlaceholderText("Search tail, callsign, type, operator, transcript…")
        self.event_search.textChanged.connect(self.render_event_history)
        self.event_filter = QComboBox()
        self.event_filter.addItems(
            [
                "All", "Confirmed", "Possible", "Pending", "Expired", "Corrected", "Cancelled",
                "Denied", "Unresolved", "Ambiguous", "Outbound filtered",
                "Airline filtered", "Duplicate", "Delivery failed",
            ]
        )
        self.event_filter.currentTextChanged.connect(self.render_event_history)
        self.event_sort = QComboBox()
        self.event_sort.addItems(
            ["Newest first", "Oldest first", "Tail number", "Aircraft type", "Decision"]
        )
        self.event_sort.currentTextChanged.connect(self.render_event_history)
        controls.addWidget(self.event_search, 1)
        controls.addWidget(self.event_filter)
        controls.addWidget(self.event_sort)
        controls.addWidget(self._button("Export", self.export_event_history))
        controls.addWidget(self._button("Refresh", self.refresh_event_history))
        layout.addLayout(controls)

        self.event_table = QTableWidget(0, 6)
        self.event_table.setObjectName("notificationTable")
        self.event_table.setHorizontalHeaderLabels(
            ["Time", "Tail", "Aircraft type", "Destination", "Decision", "Delivery"]
        )
        header = self.event_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.event_table.verticalHeader().setVisible(False)
        self.event_table.setAlternatingRowColors(True)
        self.event_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.event_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.event_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.event_table.itemSelectionChanged.connect(self._show_selected_event)
        layout.addWidget(self.event_table, 1)

        self.event_details = QTextEdit()
        self.event_details.setReadOnly(True)
        self.event_details.setMinimumHeight(190)
        self.event_details.setPlaceholderText("Select an event to see its complete evidence.")
        layout.addWidget(self.event_details)
        event_actions = QHBoxLayout()
        event_actions.addWidget(
            self._button("Seen", lambda: self.acknowledge_selected_event("seen", True))
        )
        event_actions.addWidget(
            self._button(
                "Aircraft Arrived",
                lambda: self.acknowledge_selected_event("aircraft_arrived", True),
            )
        )
        event_actions.addWidget(
            self._button(
                "False Detection",
                lambda: self.acknowledge_selected_event("false_detection", True),
            )
        )
        event_actions.addStretch()
        layout.addLayout(event_actions)
        return tab

    def _notification_metric(self, title: str, value: str) -> tuple[QFrame, QLabel]:
        frame = QFrame()
        frame.setObjectName("metricCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 10, 14, 10)
        label = QLabel(title)
        label.setObjectName("cardTitle")
        metric = QLabel(value)
        metric.setObjectName("metricValue")
        layout.addWidget(label)
        layout.addWidget(metric)
        return frame, metric

    def _button(self, text: str, callback: Any) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        button.setToolTip(text)
        return button

    def _shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+R"), self, self.restart_or_start)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self, self.stop_server)
        QShortcut(QKeySequence("Ctrl+L"), self, lambda: self.command_input.setFocus())
        QShortcut(QKeySequence("Ctrl+F"), self, lambda: self.search_input.setFocus())
        QShortcut(QKeySequence("Ctrl+O"), self, self.open_config)
        QShortcut(QKeySequence("Ctrl+Q"), self, lambda: self.close())

    def _wire_processes(self) -> None:
        self.backend_process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.backend_process.readyReadStandardOutput.connect(
            lambda: self._read_process(self.backend_process, False)
        )
        self.backend_process.readyReadStandardError.connect(
            lambda: self._read_process(self.backend_process, True)
        )
        self.backend_process.started.connect(self._backend_started)
        self.backend_process.finished.connect(self._backend_finished)
        self.backend_process.errorOccurred.connect(
            lambda error: self.append_log("ERROR", f"Backend process error: {error}")
        )
        self.command_process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.command_process.readyReadStandardOutput.connect(
            lambda: self._read_process(self.command_process, False)
        )
        self.command_process.readyReadStandardError.connect(
            lambda: self._read_process(self.command_process, True)
        )
        self.command_process.started.connect(self._command_started)
        self.command_process.finished.connect(self._command_finished)

    def _create_tray(self) -> None:
        self.tray = QSystemTrayIcon(self)
        self.tray.setToolTip("MRY Alert Control")
        menu = QMenu()
        show_action = QAction("Show Window", self)
        show_action.triggered.connect(self.show_normal)
        start_action = QAction("Start Server", self)
        start_action.triggered.connect(self.start_server)
        stop_action = QAction("Stop Server", self)
        stop_action.triggered.connect(self.stop_server)
        test_action = QAction("Test Alert", self)
        test_action.triggered.connect(self.test_alert)
        logs_action = QAction("Open Logs", self)
        logs_action.triggered.connect(self.open_logs)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        for action in (
            show_action,
            start_action,
            stop_action,
            test_action,
            logs_action,
            exit_action,
        ):
            menu.addAction(action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda reason: (
                self.show_normal()
                if reason == QSystemTrayIcon.ActivationReason.DoubleClick
                else None
            )
        )
        self._update_tray(IndicatorState.FAILED)
        self.tray.show()

    def _tray_icon(self, state: IndicatorState) -> QIcon:
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(COLORS[state]))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(3, 3, 26, 26)
        painter.end()
        return QIcon(pixmap)

    def _update_tray(self, state: IndicatorState) -> None:
        self.tray.setIcon(self._tray_icon(state))

    def start_server(self) -> None:
        if self.backend_process.state() != QProcess.ProcessState.NotRunning:
            return
        if self.external_backend and self.health_ok:
            QMessageBox.information(
                self, "External backend", "An external backend is already running."
            )
            return
        self.config, self.config_error = validate_config(self.config_path)
        if not self.config:
            self.append_log("ERROR", f"Configuration invalid: {self.config_error}")
            QMessageBox.critical(self, "Invalid configuration", self.config_error or "Unknown")
            self.update_status_cards()
            return
        program, arguments = backend_command(self.paths, self.config_path)
        self.append_log("INFO", f"Starting: {display_command(program, arguments)}")
        self.backend_process.setWorkingDirectory(str(self.paths.application_dir))
        self.backend_owned = True
        self.backend_process.start(program, arguments)
        self._refresh_buttons()

    def stop_server(self) -> None:
        if not self.backend_owned:
            if self.external_backend:
                QMessageBox.information(
                    self,
                    "External backend",
                    "This backend was not launched by MRY Alert Control and will not be stopped.",
                )
            return
        if self.backend_process.state() == QProcess.ProcessState.NotRunning:
            return
        self.append_log("INFO", "Requesting backend shutdown")
        self.backend_process.terminate()
        QTimer.singleShot(4000, self._force_backend_stop)

    def _force_backend_stop(self) -> None:
        if self.backend_process.state() == QProcess.ProcessState.NotRunning:
            return
        pid = int(self.backend_process.processId())
        self.append_log("WARNING", f"Backend did not stop gracefully; ending owned tree PID {pid}")
        if sys.platform == "win32" and pid:
            killer = QProcess(self)
            killer.start("taskkill", ["/PID", str(pid), "/T", "/F"])
            killer.waitForFinished(3000)
        else:
            self.backend_process.kill()

    def restart_server(self) -> None:
        if self.external_backend and not self.backend_owned:
            return
        if self.backend_process.state() == QProcess.ProcessState.NotRunning:
            self.start_server()
            return
        self.stop_server()
        self.backend_process.finished.connect(self._start_once_after_stop)

    def _start_once_after_stop(self) -> None:
        with suppress(RuntimeError):
            self.backend_process.finished.disconnect(self._start_once_after_stop)
        QTimer.singleShot(250, self.start_server)

    def restart_or_start(self) -> None:
        if self.backend_process.state() == QProcess.ProcessState.NotRunning:
            self.start_server()
        else:
            self.restart_server()

    def _sync_alert_sensitivity(self) -> None:
        configured = self.config.detection.alert_sensitivity if self.config else "conservative"
        label = {
            "conservative": "Conservative",
            "balanced": "Balanced",
            "never_miss": "Never Miss",
        }[configured]
        self.sensitivity_combo.setCurrentText(label)
        self._update_sensitivity_description(label)

    def _update_sensitivity_description(self, label: str) -> None:
        colors = {
            "Conservative": "#39c879",
            "Balanced": "#f0a43c",
            "Never Miss": "#ef6461",
        }
        descriptions = {
            "Conservative": (
                "Requires pilot-arrival evidence and a strong ADS-B match. Lowest false-alert risk."
            ),
            "Balanced": (
                "Also accepts controller routing when ADS-B clearly shows one aircraft moving "
                "toward Monterey Jet Center. Recommended for incomplete pilot audio."
            ),
            "Never Miss": (
                "Treats any exact Jet Center mention with plausible ground ADS-B traffic as a "
                "possible arrival. Highest alert coverage. "
                '<span style="color:#ef6461; font-weight:700;">'
                "Highest risk of false detections.</span>"
            ),
        }
        color = colors.get(label, "#d9e2ee")
        self.sensitivity_combo.setStyleSheet(
            f"QComboBox {{ color: {color}; font-weight: 700; }}"
        )
        self.sensitivity_description.setTextFormat(Qt.TextFormat.RichText)
        self.sensitivity_description.setText(descriptions.get(label, ""))

    def apply_alert_sensitivity(self) -> None:
        mode = {
            "Conservative": "conservative",
            "Balanced": "balanced",
            "Never Miss": "never_miss",
        }[self.sensitivity_combo.currentText()]
        try:
            update_alert_sensitivity(self.config_path, mode)
            config, error = validate_config(self.config_path)
            if config is None:
                raise ValueError(error or "configuration validation failed")
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Alert sensitivity", f"Could not save setting: {exc}")
            return
        self.config = config
        self.config_error = None
        self.append_log("INFO", f"Alert sensitivity changed to {mode}")
        if self.backend_owned and self.backend_process.state() != QProcess.ProcessState.NotRunning:
            self.restart_server()
            self.statusBar().showMessage(f"Applying {mode} mode and restarting backend", 5000)
        elif self.health_ok:
            QMessageBox.information(
                self,
                "Restart required",
                "The setting was saved. Restart the externally launched backend to apply it.",
            )
        else:
            self.statusBar().showMessage(
                f"{mode.replace('_', ' ').title()} mode saved for the next backend start",
                5000,
            )

    def _backend_started(self) -> None:
        self.append_log("INFO", "Backend process started")
        self.tray.showMessage("MRY Alert Control", "Backend started")
        self._refresh_buttons()

    def _backend_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        unexpected = exit_code != 0
        level = "ERROR" if unexpected else "INFO"
        self.append_log(level, f"Backend exited with code {exit_code} ({exit_status.name})")
        if unexpected:
            self.tray.showMessage(
                "MRY Alert Control", f"Backend stopped unexpectedly ({exit_code})"
            )
        self.backend_owned = False
        self.health_ok = False
        self._refresh_buttons()
        self.update_status_cards()

    def run_manual_command(self) -> None:
        if self.command_process.state() != QProcess.ProcessState.NotRunning:
            return
        raw = self.command_input.text()
        try:
            arguments = parse_safe_command(raw)
        except ValueError as exc:
            self.append_log("ERROR", str(exc))
            self.statusBar().showMessage(str(exc), 5000)
            return
        self.command_history.append(raw)
        self.history_index = len(self.command_history)
        program, process_arguments = manual_command(self.paths, arguments)
        self.append_log("INFO", f"Command: {display_command(program, process_arguments)}")
        self.command_process.setWorkingDirectory(str(self.paths.application_dir))
        self.command_process.start(program, process_arguments)

    def cancel_manual_command(self) -> None:
        if self.command_process.state() == QProcess.ProcessState.NotRunning:
            return
        self.append_log("WARNING", "Cancelling manual command")
        self.command_process.terminate()
        QTimer.singleShot(
            2500,
            lambda: (
                self.command_process.kill()
                if self.command_process.state() != QProcess.ProcessState.NotRunning
                else None
            ),
        )

    def _command_started(self) -> None:
        self._refresh_buttons()

    def _command_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        level = "INFO" if exit_code == 0 else "ERROR"
        self.append_log(level, f"Manual command exited with code {exit_code} ({exit_status.name})")
        self._refresh_buttons()
        self.update_training_tab()

    def navigate_history(self, delta: int) -> None:
        if not self.command_history:
            return
        self.history_index = max(0, min(len(self.command_history), self.history_index + delta))
        value = (
            self.command_history[self.history_index]
            if self.history_index < len(self.command_history)
            else ""
        )
        self.command_input.setText(value)

    def _read_process(self, process: QProcess, stderr: bool) -> None:
        data = process.readAllStandardError() if stderr else process.readAllStandardOutput()
        text = bytes(data.data()).decode("utf-8", errors="replace")
        default_level = "ERROR" if stderr else "INFO"
        for line in text.splitlines():
            level = self._classify_line(line, default_level)
            self.append_log(level, line)
            if process is self.command_process and hasattr(self, "training_output"):
                self.training_output.append(line)

    @staticmethod
    def _classify_line(line: str, default: str) -> str:
        upper = line.strip().upper()
        if upper.startswith(("WARNING:", "WARN:")):
            return "WARNING"
        if upper.startswith(("ERROR:", "CRITICAL:")):
            return "ERROR"
        if "ADS-B" in upper:
            return "ADS-B"
        if "NOTIFICATION" in upper:
            return "NOTIFICATION"
        if "TRANSCRIPT" in upper or "HEARD:" in upper:
            return "TRANSCRIPTION"
        if upper.startswith(("INFO:", "DEBUG:")):
            return "INFO"
        if "ERROR" in upper or "FAILED" in upper or "REJECTED" in upper:
            return "ERROR"
        if "WARNING" in upper or "AMBIGUOUS" in upper or "UNRESOLVED" in upper:
            return "WARNING"
        return default

    def append_log(self, category: str, message: str) -> None:
        safe = redact_secrets(message)
        timestamped = f"{datetime.now():%Y-%m-%d %H:%M:%S} {category} {safe}"
        self.log_lines.append((category, timestamped))
        self.paths.gui_logs_dir.mkdir(parents=True, exist_ok=True)
        with self.session_log.open("a", encoding="utf-8") as handle:
            handle.write(timestamped + "\n")
        if not self.display_paused and self._line_visible(category, timestamped):
            self._append_colored(category, timestamped)

    def _line_visible(self, category: str, line: str) -> bool:
        selected = self.log_filter.currentText()
        search = self.search_input.text().lower()
        return (selected == "ALL" or category == selected) and (
            not search or search in line.lower()
        )

    def _append_colored(self, category: str, line: str) -> None:
        color = {
            "ERROR": "#ef6461",
            "WARNING": "#f0b44d",
            "NOTIFICATION": "#39c879",
        }.get(category, "#d7dce3")
        self.log_view.append(
            f'<span style="color:{color}; white-space:pre">{html.escape(line)}</span>'
        )
        if self.autoscroll.isChecked():
            self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def rebuild_logs(self) -> None:
        if self.display_paused:
            return
        self.log_view.clear()
        for category, line in self.log_lines:
            if self._line_visible(category, line):
                self._append_colored(category, line)

    def _set_pause(self, paused: bool) -> None:
        self.display_paused = paused
        if not paused:
            self.rebuild_logs()

    def clear_log_display(self) -> None:
        self.log_view.clear()

    def clear_log_files(self) -> None:
        answer = QMessageBox.question(
            self,
            "Clear GUI logs",
            "Delete saved GUI session logs? Backend event history will not be deleted.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        for path in self.paths.gui_logs_dir.glob("*.log"):
            if path != self.session_log:
                path.unlink(missing_ok=True)
        self.session_log.write_text("", encoding="utf-8")
        self.log_lines.clear()
        self.log_view.clear()
        self.append_log("INFO", "GUI session logs cleared")

    def copy_logs(self) -> None:
        cursor = self.log_view.textCursor()
        QApplication.clipboard().setText(
            cursor.selectedText() if cursor.hasSelection() else self.log_view.toPlainText()
        )

    def save_logs(self) -> None:
        destination, _ = QFileDialog.getSaveFileName(
            self, "Save redacted logs", str(self.paths.logs_dir / "mry-control.log"), "Log (*.log)"
        )
        if destination:
            Path(destination).write_text(
                "\n".join(line for _, line in self.log_lines) + "\n",
                encoding="utf-8",
            )

    def poll_backend(self) -> None:
        if self._health_reply and self._health_reply.isRunning():
            return
        request = QNetworkRequest(QUrl(f"{self.base_url}/health"))
        self._health_reply = self.network.get(request)
        self._health_reply.finished.connect(self._health_finished)

    def _health_finished(self) -> None:
        reply = self._health_reply
        if reply is None:
            return
        self.health_ok = (
            reply.error() == QNetworkReply.NetworkError.NoError
            and reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute) == 200
        )
        process_running = self.backend_process.state() != QProcess.ProcessState.NotRunning
        self.external_backend = self.health_ok and not process_running
        reply.deleteLater()
        self._health_reply = None
        if self.health_ok:
            status_request = QNetworkRequest(QUrl(f"{self.base_url}/api/status"))
            self._status_reply = self.network.get(status_request)
            self._status_reply.finished.connect(self._status_finished)
            self.refresh_notifications()
            self.refresh_event_history()
        else:
            self.status_payload = None
            self.backend_session_id = None
            self.event_history_records = []
            self.render_event_history()
            self.update_status_cards()
            self._refresh_buttons()

    def _status_finished(self) -> None:
        reply = self._status_reply
        if reply is None:
            return
        if reply.error() == QNetworkReply.NetworkError.NoError:
            import json

            try:
                value = json.loads(bytes(reply.readAll().data()).decode("utf-8"))
                self.status_payload = value if isinstance(value, dict) else None
            except (UnicodeDecodeError, ValueError):
                self.status_payload = None
        else:
            self.status_payload = None
        reply.deleteLater()
        self._status_reply = None
        self.update_status_cards()
        self.update_ntfy_details()
        self.update_training_tab()
        self.update_activity()
        self._refresh_buttons()

    def refresh_notifications(self) -> None:
        if not self.health_ok:
            self.statusBar().showMessage("Start the backend to refresh notification history", 4000)
            return
        if self._notifications_reply and self._notifications_reply.isRunning():
            return
        request = QNetworkRequest(QUrl(f"{self.base_url}/api/notifications"))
        self._notifications_reply = self.network.get(request)
        self._notifications_reply.finished.connect(self._notifications_finished)

    def _notifications_finished(self) -> None:
        reply = self._notifications_reply
        if reply is None:
            return
        if reply.error() == QNetworkReply.NetworkError.NoError:
            try:
                value = json.loads(bytes(reply.readAll().data()).decode("utf-8"))
                if isinstance(value, list):
                    self.notification_records = [item for item in value if isinstance(item, dict)]
            except (UnicodeDecodeError, ValueError):
                self.append_log("WARNING", "Could not read notification history response")
        reply.deleteLater()
        self._notifications_reply = None
        self.render_notification_history()

    def refresh_event_history(self) -> None:
        if not self.health_ok or not self.config:
            return
        if self._history_reply and self._history_reply.isRunning():
            return
        try:
            token = read_pairing_token(self.config, self.config_path)
        except (OSError, ValueError) as exc:
            self.append_log("WARNING", f"Could not authenticate event history: {exc}")
            return
        request = QNetworkRequest(QUrl(f"{self.base_url}/api/event-history"))
        request.setRawHeader(b"X-Pairing-Token", token.encode("utf-8"))
        self._history_reply = self.network.get(request)
        self._history_reply.finished.connect(self._history_finished)

    def _history_finished(self) -> None:
        reply = self._history_reply
        if reply is None:
            return
        if reply.error() == QNetworkReply.NetworkError.NoError:
            try:
                value = json.loads(bytes(reply.readAll().data()).decode("utf-8"))
                if isinstance(value, dict):
                    session_id = value.get("session_id")
                    events = value.get("events")
                    if isinstance(session_id, str) and isinstance(events, list):
                        if self.backend_session_id and session_id != self.backend_session_id:
                            self.event_history_records = []
                        self.backend_session_id = session_id
                        self.event_history_records = [
                            item for item in events if isinstance(item, dict)
                        ]
            except (UnicodeDecodeError, ValueError):
                self.append_log("WARNING", "Could not read session event history response")
        reply.deleteLater()
        self._history_reply = None
        self.render_event_history()

    def _displayed_event_history(self) -> list[dict[str, Any]]:
        return filter_session_events(
            self.event_history_records,
            self.event_search.text(),
            self.event_filter.currentText(),
            self.event_sort.currentText(),
        )

    @staticmethod
    def _event_delivery(record: dict[str, Any]) -> str:
        chrome = str(record.get("chrome_delivery_result") or "not attempted")
        ntfy = str(record.get("ntfy_delivery_result") or "not attempted")
        if any("fail" in value for value in (chrome.casefold(), ntfy.casefold())):
            return "Failed / partial"
        if chrome == "delivered" and ntfy == "delivered":
            return "Chrome + ntfy"
        if chrome == "delivered":
            return "Chrome delivered"
        if ntfy == "delivered":
            return "ntfy delivered"
        return str(record.get("notification_status") or "Not sent").replace("_", " ").title()

    def render_event_history(self) -> None:
        if not hasattr(self, "event_table"):
            return
        selected_key: tuple[str, str] | None = None
        selected_row = self.event_table.currentRow()
        selected_item = self.event_table.item(selected_row, 0) if selected_row >= 0 else None
        selected_record = (
            selected_item.data(Qt.ItemDataRole.UserRole) if selected_item else None
        )
        if isinstance(selected_record, dict):
            selected_key = (
                str(selected_record.get("event_id") or ""),
                str(selected_record.get("transition_type") or ""),
            )
        scroll_position = self.event_table.verticalScrollBar().value()
        records = self._displayed_event_history()
        self.event_table.setRowCount(0)
        restored_row: int | None = None
        for row, record in enumerate(records):
            self.event_table.insertRow(row)
            aircraft_type = (
                record.get("aircraft_type_name")
                or record.get("aircraft_type")
                or "Unknown"
            )
            values = (
                self._display_timestamp(record.get("timestamp")),
                record.get("registration") or "Unresolved",
                aircraft_type,
                record.get("destination") or "-",
                str(record.get("transition_type") or record.get("final_decision") or "-")
                .replace("_", " ")
                .title(),
                self._event_delivery(record),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, record)
                    record_key = (
                        str(record.get("event_id") or ""),
                        str(record.get("transition_type") or ""),
                    )
                    if selected_key is not None and record_key == selected_key:
                        restored_row = row
                if column == 5 and "fail" in str(value).casefold():
                    item.setForeground(QColor("#ef6461"))
                self.event_table.setItem(row, column, item)
            self.event_table.setRowHeight(row, 38)
        if restored_row is not None:
            self.event_table.selectRow(restored_row)
            self.event_table.verticalScrollBar().setValue(scroll_position)
        elif records:
            self.event_table.selectRow(0)
        else:
            self.event_details.clear()
            self.event_details.setPlaceholderText("No session events match this view.")

    def _show_selected_event(self) -> None:
        row = self.event_table.currentRow()
        item = self.event_table.item(row, 0) if row >= 0 else None
        record = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not isinstance(record, dict):
            return
        reasons = record.get("decision_reasons") or []
        evidence = "\n".join(f"• {reason}" for reason in reasons) or "• None recorded"
        def confidence(value: object) -> str:
            return f"{float(value):.0%}" if isinstance(value, int | float) else "unknown"

        aircraft_type = (
            record.get("aircraft_type_name") or record.get("aircraft_type") or "Unknown"
        )
        score = record.get("adsb_score")
        margin = record.get("winning_margin")
        acknowledgement = str(
            record.get("operator_acknowledgement") or "unreviewed"
        ).replace("_", " ").title()
        lines = [
            f"Event ID: {record.get('event_id') or '-'}",
            f"Related/original event: {record.get('original_event_id') or '-'}",
            f"Tail: {record.get('registration') or 'unresolved'}   "
            f"Spoken clue: {record.get('spoken_callsign') or 'none'}",
            f"Aircraft: {aircraft_type}   "
            f"ICAO type: {record.get('aircraft_type_code') or 'unknown'}",
            f"Manufacturer/model: {record.get('manufacturer') or '-'} / "
            f"{record.get('model') or '-'}",
            f"Category: {record.get('aircraft_category') or '-'}   "
            f"Operator: {record.get('operator_name') or '-'}",
            f"Type source/confidence: {record.get('aircraft_type_source') or 'unknown'} / "
            f"{confidence(record.get('aircraft_type_confidence'))}",
            f"Destination: {record.get('destination') or '-'}   "
            f"Intent: {record.get('intent') or '-'}",
            f"Direction: {record.get('direction_state') or '-'}   "
            f"Movement: {record.get('adsb_movement_state') or '-'}",
            f"ADS-B score: {score if score is not None else '-'}   "
            f"Winning margin: {margin if margin is not None else '-'}",
            f"Decision: {record.get('final_decision') or '-'}   "
            f"Notification: {record.get('notification_status') or '-'}",
            f"Chrome: {record.get('chrome_delivery_result') or '-'}   "
            f"ntfy: {record.get('ntfy_delivery_result') or '-'}",
            f"Operator acknowledgement: {acknowledgement}",
            f"Acknowledged at: {self._display_timestamp(record.get('acknowledged_at'))}",
            f"Classifier: {confidence(record.get('classifier_confidence'))}   "
            f"Whisper: {confidence(record.get('whisper_confidence'))}   "
            f"Decoder: {record.get('decoder_confidence') or 'unknown'}",
            f"Transcript: “{record.get('transcript_excerpt') or 'Not available'}”",
            f"Decision evidence:\n{evidence}",
        ]
        self.event_details.setPlainText("\n".join(lines))

    def export_event_history(self) -> None:
        records = self._displayed_event_history()
        if not records:
            QMessageBox.information(self, "Export Event History", "There are no displayed events.")
            return
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Export displayed session history",
            str(self.paths.data_dir / "session-event-history.json"),
            "JSON (*.json)",
        )
        if destination:
            Path(destination).write_text(
                json.dumps(
                    {"session_id": self.backend_session_id, "events": records},
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

    def _filtered_notifications(self) -> list[dict[str, Any]]:
        selected = self.notification_filter.currentText()
        if selected == "Real alerts":
            return [item for item in self.notification_records if not item.get("test")]
        if selected == "Test alerts":
            return [item for item in self.notification_records if item.get("test")]
        return self.notification_records

    @staticmethod
    def _display_timestamp(value: object) -> str:
        if not isinstance(value, str):
            return "-"
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.astimezone().strftime("%b %d  %I:%M:%S %p")
        except ValueError:
            return value

    def render_notification_history(self) -> None:
        records = self._filtered_notifications()
        selected_event_id: str | None = None
        selected_row = self.notification_table.currentRow()
        selected_item = (
            self.notification_table.item(selected_row, 0) if selected_row >= 0 else None
        )
        selected_record = (
            selected_item.data(Qt.ItemDataRole.UserRole) if selected_item else None
        )
        if isinstance(selected_record, dict):
            selected_event_id = str(selected_record.get("event_id") or "")
        self.notification_table.setRowCount(0)
        delivered_records = sum(int(item.get("delivered_clients", 0)) > 0 for item in records)
        self.notification_total.setText(str(len(records)))
        self.notification_delivered.setText(str(delivered_records))
        self.notification_last.setText(
            self._display_timestamp(records[0].get("sent_at")) if records else "None yet"
        )

        restored_row: int | None = None
        for row, record in enumerate(records):
            self.notification_table.insertRow(row)
            delivered = int(record.get("delivered_clients", 0))
            connected = int(record.get("connected_clients", 0))
            failed = int(record.get("failed_clients", 0))
            if delivered:
                delivery = f"Delivered to {delivered}"
            elif connected or failed:
                delivery = "Delivery failed"
            else:
                delivery = "No extension connected"
            confidence = record.get("confidence")
            confidence_text = (
                f"{float(confidence):.0%}" if isinstance(confidence, int | float) else "-"
            )
            values = (
                self._display_timestamp(record.get("sent_at")),
                record.get("registration") or record.get("spoken_callsign") or "Unresolved",
                record.get("destination") or "-",
                "Test" if record.get("test") else "Aircraft alert",
                confidence_text,
                delivery,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, record)
                    if selected_event_id == str(record.get("event_id") or ""):
                        restored_row = row
                if column == 5:
                    item.setForeground(QColor("#39c879" if delivered else "#f0b44d"))
                self.notification_table.setItem(row, column, item)
            self.notification_table.setRowHeight(row, 38)

        if restored_row is not None:
            self.notification_table.selectRow(restored_row)
        elif records:
            self.notification_table.selectRow(0)
        else:
            self.notification_details.clear()
            self.notification_details.setPlaceholderText("No notifications match this view yet.")

    def _show_selected_notification(self) -> None:
        row = self.notification_table.currentRow()
        if row < 0:
            return
        item = self.notification_table.item(row, 0)
        record = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not isinstance(record, dict):
            return
        reasons = record.get("match_reasons") or []
        reason_text = "\n".join(f"• {reason}" for reason in reasons) or "• None recorded"
        transcript = record.get("transcript_excerpt") or "Not available"
        aircraft_type = record.get("aircraft_type") or "Unknown aircraft type"
        acknowledgement = str(
            record.get("operator_acknowledgement") or "unreviewed"
        ).replace("_", " ").title()
        self.notification_details.setPlainText(
            f"{record.get('registration') or record.get('spoken_callsign') or 'Unresolved'}"
            f"  ·  {aircraft_type}\n"
            f"Heard: “{transcript}”\n"
            f"Why it matched:\n{reason_text}\n"
            f"Operator acknowledgement: {acknowledgement}"
        )

    def acknowledge_selected_event(self, acknowledgement: str, history: bool) -> None:
        if not self.health_ok or not self.config:
            self.statusBar().showMessage("Start the backend before recording an outcome", 4000)
            return
        table = self.event_table if history else self.notification_table
        row = table.currentRow()
        item = table.item(row, 0) if row >= 0 else None
        record = item.data(Qt.ItemDataRole.UserRole) if item else None
        event_id = str(record.get("event_id") or "") if isinstance(record, dict) else ""
        if not event_id:
            self.statusBar().showMessage("Select an event first", 3000)
            return
        if self._acknowledgement_reply and self._acknowledgement_reply.isRunning():
            return
        try:
            token = read_pairing_token(self.config, self.config_path)
        except (OSError, ValueError) as exc:
            self.append_log("WARNING", f"Could not authenticate acknowledgement: {exc}")
            return
        request = QNetworkRequest(
            QUrl(f"{self.base_url}/api/event-history/{event_id}/acknowledgement")
        )
        request.setRawHeader(b"X-Pairing-Token", token.encode("utf-8"))
        request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json")
        payload = QByteArray(json.dumps({"acknowledgement": acknowledgement}).encode("utf-8"))
        self._acknowledgement_reply = self.network.post(request, payload)
        self._acknowledgement_reply.finished.connect(self._acknowledgement_finished)

    def _acknowledgement_finished(self) -> None:
        reply = self._acknowledgement_reply
        if reply is None:
            return
        if reply.error() == QNetworkReply.NetworkError.NoError:
            try:
                record = json.loads(bytes(reply.readAll().data()).decode("utf-8"))
                event_id = str(record.get("event_id") or "")
                self.event_history_records = [
                    record if str(item.get("event_id") or "") == event_id else item
                    for item in self.event_history_records
                ]
                self.notification_records = [
                    {
                        **item,
                        "operator_acknowledgement": record.get(
                            "operator_acknowledgement", "unreviewed"
                        ),
                        "acknowledged_at": record.get("acknowledged_at"),
                    }
                    if str(item.get("event_id") or "") == event_id
                    else item
                    for item in self.notification_records
                ]
                acknowledgement = str(record.get("operator_acknowledgement") or "")
                self.statusBar().showMessage(
                    f"Outcome recorded: {acknowledgement.replace('_', ' ').title()}", 4000
                )
                self.render_event_history()
                self._show_selected_notification()
            except (UnicodeDecodeError, ValueError, AttributeError):
                self.append_log("WARNING", "Could not read acknowledgement response")
        else:
            code = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
            message = (
                "That notification belongs to a previous backend session"
                if code == 404
                else "Could not record the selected outcome"
            )
            self.statusBar().showMessage(message, 5000)
            self.append_log("WARNING", message)
        reply.deleteLater()
        self._acknowledgement_reply = None

    def update_status_cards(self) -> None:
        self.config, self.config_error = validate_config(self.config_path)
        token_found = False
        if self.config:
            token_found = pairing_token_path(self.config, self.config_path).is_file()
        indicators = status_indicators(
            self.status_payload,
            process_running=self.backend_process.state() != QProcess.ProcessState.NotRunning,
            health_ok=self.health_ok,
            external=self.external_backend,
            config_ok=self.config is not None,
            token_found=token_found,
        )
        for name, card in self.cards.items():
            card.update_indicator(indicators[name])
        self.update_training_tab()
        critical = indicators["Backend server"].state
        if critical == IndicatorState.HEALTHY:
            degraded = any(
                item.state in {IndicatorState.DEGRADED, IndicatorState.FAILED}
                for key, item in indicators.items()
                if key not in {"Last transcription", "Last ADS-B correlation"}
            )
            critical = IndicatorState.DEGRADED if degraded else IndicatorState.HEALTHY
        self._update_tray(critical)

    def update_ntfy_details(self) -> None:
        value = self.status_payload or {}
        if not value.get("ntfy_enabled"):
            self.ntfy_details.setText(
                "Disabled. Configure server URL, topic, and optional Authorization header "
                "under ntfy in config.yaml, then restart the backend."
            )
            return
        authorization = (
            "configured (hidden)" if value.get("ntfy_authorization_configured") else "not used"
        )
        last_result = value.get("ntfy_last_success")
        result_text = (
            "delivered successfully"
            if last_result is True
            else f"failed — {value.get('ntfy_last_error') or 'unknown error'}"
            if last_result is False
            else "no push attempted yet"
        )
        self.ntfy_details.setText(
            f"Server: {value.get('ntfy_server_url') or '-'}\n"
            f"Subscribed topic: {value.get('ntfy_topic') or '-'}\n"
            f"Subscription URL: {value.get('ntfy_subscription_url') or '-'}\n"
            f"Authorization: {authorization}\n"
            f"Last push: {result_text}"
        )

    def update_activity(self) -> None:
        value = self.status_payload or {}
        transcript = value.get("last_transcript")
        if not transcript:
            self.activity_label.setText("No qualifying event yet")
            return
        score = value.get("last_adsb_score")
        margin = value.get("last_adsb_margin")
        delivered = value.get("last_notification_delivered", 0)
        self.activity_label.setText(
            f"Heard: “{transcript}”\n"
            f"Callsign: {value.get('last_detected_callsign') or 'unresolved'}\n"
            f"Aircraft type: {value.get('last_aircraft_type') or 'unknown'}\n"
            f"Destination: {value.get('last_destination') or 'none'}\n"
            f"ADS-B: {value.get('last_adsb_winner') or 'unresolved'} — "
            f"score {score if score is not None else '-'}, "
            f"margin {margin if margin is not None else '-'}\n"
            f"Decision: {value.get('last_detector_decision') or 'none'}\n"
            f"Notification: sent to {delivered} extension client(s)\n"
            f"Timestamp: {value.get('last_transcription_at') or '-'}"
        )

    def test_alert(self) -> None:
        if not self.health_ok or not self.config:
            QMessageBox.warning(self, "Test Alert", "Backend health is unavailable.")
            return
        try:
            token = read_pairing_token(self.config, self.config_path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Pairing token", str(exc))
            return
        request = QNetworkRequest(QUrl(f"{self.base_url}/api/test-alert"))
        request.setRawHeader(b"X-Pairing-Token", token.encode("utf-8"))
        reply = self.network.post(request, QByteArray())

        def complete() -> None:
            if reply.error() == QNetworkReply.NetworkError.NoError:
                self.append_log("NOTIFICATION", "Authenticated test alert requested successfully")
                self.statusBar().showMessage("Test alert sent", 5000)
                self.refresh_notifications()
            else:
                self.append_log("ERROR", f"Test alert failed: {reply.errorString()}")
                self.tray.showMessage("MRY Alert Control", "Test alert failed")
            reply.deleteLater()

        reply.finished.connect(complete)

    def train_classifier(self) -> None:
        if self.command_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Training", "Another manual command is running.")
            return
        if not self.config:
            QMessageBox.warning(self, "Training", "Load a valid configuration first.")
            return
        dataset = self._configured_path(self.config.training_data.directory) / "reviewed"
        model = self._configured_path(self.config.audio_classifier.model_path)
        self.command_input.setText(
            f'train-audio-classifier --dataset "{dataset}" --output "{model}" '
            f'--config "{self.config_path}"'
        )
        self.training_output.clear()
        self.training_output.append("Starting classifier training…")
        self.run_manual_command()
        self.statusBar().showMessage("Classifier training started", 5000)

    def evaluate_classifier(self) -> None:
        if self.command_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Evaluation", "Another manual command is running.")
            return
        if not self.config:
            QMessageBox.warning(self, "Evaluation", "Load a valid configuration first.")
            return
        dataset = self._configured_path(self.config.training_data.directory) / "reviewed"
        model = self._configured_path(self.config.audio_classifier.model_path)
        self.command_input.setText(
            f'evaluate-audio-classifier --dataset "{dataset}" --model "{model}" '
            f'--config "{self.config_path}"'
        )
        self.training_output.clear()
        self.training_output.append("Starting offline evaluation…")
        self.run_manual_command()
        self.statusBar().showMessage("Offline classifier evaluation started", 5000)

    def _configured_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.config_path.parent / path

    def refresh_training_status(self) -> None:
        self.update_training_tab()
        if self.health_ok:
            self.poll_backend()

    def update_training_tab(self) -> None:
        if not hasattr(self, "training_details"):
            return
        if not self.config:
            self.training_collection_value.setText("Config invalid")
            self.training_pending_value.setText("-")
            self.training_classifier_value.setText("Unavailable")
            self.training_details.setText(self.config_error or "Load a valid configuration.")
            self.clip_recording_button.setEnabled(False)
            return
        root = self._configured_path(self.config.training_data.directory)
        pending = scan_clips(root / "pending")
        reviewed = scan_clips(root / "reviewed", require_reviewed=True)
        hard_negatives = scan_clips(root / "hard_negatives", require_reviewed=True)
        issue_count = len(pending.issues) + len(reviewed.issues) + len(hard_negatives.issues)
        status = self.status_payload or {}
        collecting = bool(
            status.get("dataset_collection_enabled", self.config.training_data.enabled)
        )
        self.training_collection_value.setText("Enabled" if collecting else "Disabled")
        self.clip_recording_button.setEnabled(True)
        self.clip_recording_button.setText(
            "Stop Recording Clips" if self.config.training_data.enabled else "Start Recording Clips"
        )
        color = "#ef6461" if self.config.training_data.enabled else "#39c879"
        self.clip_recording_button.setStyleSheet(
            f"QPushButton {{ color: {color}; font-weight: 700; }}"
        )
        self.training_pending_value.setText(str(len(pending.valid_metadata)))
        classifier_error = status.get("classifier_error")
        classifier_loaded = bool(status.get("classifier_loaded"))
        classifier_enabled = bool(
            status.get("classifier_enabled", self.config.audio_classifier.enabled)
        )
        classifier_text = (
            "Error"
            if classifier_error
            else "Loaded"
            if classifier_loaded
            else "Not loaded"
            if classifier_enabled
            else "Disabled"
        )
        self.training_classifier_value.setText(classifier_text)
        collection_error = status.get("training_clip_last_error")
        self.training_details.setText(
            f"Clip directory: {root}\n"
            "Audio saving: "
            f"{'enabled' if self.config.training_data.save_audio else 'disabled'}  ·  "
            f"Reviewed: {len(reviewed.valid_metadata)}  ·  "
            f"Approved hard negatives: {len(hard_negatives.valid_metadata)}  ·  "
            f"Invalid items retained: {issue_count}"
            + (f"\nLast collection error: {collection_error}" if collection_error else "")
            + (f"\nClassifier error: {classifier_error}" if classifier_error else "")
        )

    def toggle_clip_recording(self) -> None:
        if not self.config:
            QMessageBox.critical(
                self,
                "Clip recording",
                "Fix the configuration before changing clip recording.",
            )
            return
        if self.external_backend and not self.backend_owned:
            QMessageBox.information(
                self,
                "External backend running",
                "Stop the externally launched backend first, then press this button again so "
                "the complete application can restart safely.",
            )
            return
        enabled = not self.config.training_data.enabled
        try:
            update_training_collection(self.config_path, enabled)
            config, error = validate_config(self.config_path)
            if config is None:
                raise ValueError(error or "configuration validation failed")
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Clip recording", f"Could not save setting: {exc}")
            return
        self.config = config
        self.config_error = None
        state = "enabled" if enabled else "disabled"
        self.append_log("INFO", f"Automatic training-clip collection {state}")
        self.statusBar().showMessage(
            f"Clip recording {state}; restarting backend and Control app", 5000
        )
        self._restart_entire_application()

    def _restart_entire_application(self) -> None:
        self._full_relaunch_in_progress = True
        if self.backend_owned and self.backend_process.state() != QProcess.ProcessState.NotRunning:
            self.backend_process.terminate()
            if not self.backend_process.waitForFinished(4500):
                self._force_backend_stop()
                self.backend_process.waitForFinished(3500)
        launcher = QProcess()
        environment = launcher.processEnvironment()
        if environment.isEmpty():
            environment = QProcessEnvironment.systemEnvironment()
        environment.insert("MRY_ALERT_RESTART_BACKEND", "1")
        launcher.setProcessEnvironment(environment)
        if bool(getattr(sys, "frozen", False)):
            launcher.setProgram(sys.executable)
            launcher.setArguments([])
        else:
            launcher.setProgram(sys.executable)
            launcher.setArguments(["-m", "mry_alert.control"])
        launcher.setWorkingDirectory(str(self.paths.application_dir))
        started, _process_id = launcher.startDetached()
        if not started:
            self._full_relaunch_in_progress = False
            QMessageBox.critical(
                self,
                "Restart failed",
                "The setting was saved, but the Control app could not relaunch. Start it again "
                "manually; the backend is currently stopped.",
            )
            return
        self.close()

    def open_training_folder(self) -> None:
        if not self.config:
            return
        directory = self._configured_path(self.config.training_data.directory)
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def clear_training_output(self) -> None:
        self.training_output.clear()

    def open_model_folder(self) -> None:
        if not self.config:
            return
        directory = self._configured_path(self.config.audio_classifier.model_path)
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def open_review_queue(self) -> None:
        if not self.config:
            QMessageBox.warning(self, "Review queue", "Load a valid configuration first.")
            return
        dialog = ReviewDialog(self._configured_path(self.config.training_data.directory), self)
        dialog.exec()
        self.update_training_tab()

    def open_config(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.config_path)))

    def select_config(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Select configuration", str(self.config_path.parent), "YAML (*.yaml *.yml)"
        )
        if not selected:
            return
        candidate = Path(selected)
        config, error = validate_config(candidate)
        if not config:
            QMessageBox.critical(self, "Invalid configuration", error or "Unknown error")
            return
        self.config_path = candidate
        self.config = config
        self.config_error = None
        self.config_label.setText(str(candidate))
        self._sync_alert_sensitivity()
        self.append_log("INFO", f"Active configuration selected: {candidate}")
        self.poll_backend()
        self.update_status_cards()
        self.update_training_tab()

    def open_logs(self) -> None:
        self.paths.gui_logs_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.paths.logs_dir)))

    def open_token_folder(self) -> None:
        if not self.config:
            return
        path = pairing_token_path(self.config, self.config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))

    def copy_pairing_token(self) -> None:
        if not self.config:
            self.statusBar().showMessage("Fix the configuration before copying the token", 5000)
            return
        try:
            token = read_pairing_token(self.config, self.config_path)
        except (OSError, ValueError) as exc:
            self.statusBar().showMessage(f"Could not copy pairing token: {exc}", 5000)
            return
        QApplication.clipboard().setText(token)
        self.statusBar().showMessage(
            "Pairing token copied. Paste it into the Chrome extension.", 5000
        )

    def toggle_startup(self, enabled: bool) -> None:
        if enabled:
            answer = QMessageBox.question(
                self,
                "Start with Windows",
                "Start MRY Alert Control automatically when you sign in to Windows?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.startup_checkbox.blockSignals(True)
                self.startup_checkbox.setChecked(False)
                self.startup_checkbox.blockSignals(False)
                return
        try:
            executable = Path(sys.executable)
            arguments: list[str] = []
            if not bool(getattr(sys, "frozen", False)):
                pythonw = executable.with_name("pythonw.exe")
                executable = pythonw if pythonw.exists() else executable
                arguments = ["-m", "mry_alert.control"]
            set_windows_startup(enabled, executable, arguments)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Windows startup", str(exc))

    def _refresh_buttons(self) -> None:
        running = self.backend_process.state() != QProcess.ProcessState.NotRunning
        self.start_button.setEnabled(not running and not self.external_backend)
        self.stop_button.setEnabled(running and self.backend_owned)
        self.restart_button.setEnabled(running and self.backend_owned)
        self.test_button.setEnabled(self.health_ok)
        command_running = self.command_process.state() != QProcess.ProcessState.NotRunning
        self.run_button.setEnabled(not command_running)
        self.cancel_button.setEnabled(command_running)
        if hasattr(self, "train_button"):
            self.train_button.setEnabled(not command_running)
            self.evaluate_button.setEnabled(not command_running)
            self.review_button.setEnabled(not command_running and self.config is not None)
            self.clip_recording_button.setEnabled(
                not command_running and self.config is not None
            )

    def show_normal(self) -> None:
        self.show()
        self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
        self.raise_()
        self.activateWindow()

    def changeEvent(self, event: QEvent) -> None:
        if (
            event.type() == QEvent.Type.WindowStateChange
            and self.isMinimized()
            and self.minimize_checkbox.isChecked()
        ):
            QTimer.singleShot(0, self.hide)
            self.tray.showMessage("MRY Alert Control", "Still running in the system tray")
        super().changeEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._full_relaunch_in_progress:
            if self.backend_process.state() != QProcess.ProcessState.NotRunning:
                self.backend_process.kill()
                self.backend_process.waitForFinished(3000)
            self.settings.setValue("geometry", self.saveGeometry())
            self.settings.setValue("minimize_to_tray", self.minimize_checkbox.isChecked())
            self.tray.hide()
            event.accept()
            return
        if self.backend_owned and self.backend_process.state() != QProcess.ProcessState.NotRunning:
            answer = QMessageBox.question(
                self,
                "Exit MRY Alert Control",
                "Stop the backend launched by this application before exiting?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.StandardButton.Yes:
                self.stop_server()
                self.backend_process.waitForFinished(4500)
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("minimize_to_tray", self.minimize_checkbox.isChecked())
        self.tray.hide()
        event.accept()

    def _restore_geometry(self) -> None:
        geometry = self.settings.value("geometry")
        if isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)

    @staticmethod
    def _stylesheet() -> str:
        return """
        QMainWindow, QWidget { background: #161a20; color: #d7dce3; }
        QLabel { background: transparent; }
        QLabel#mainTitle { font-size: 25px; font-weight: 700; color: #f2f5f8; }
        QLabel#sectionTitle { font-size: 23px; font-weight: 700; color: #f2f5f8; }
        QLabel#subtitle { color: #8f99a8; }
        QLabel#metricValue { font-size: 20px; font-weight: 700; color: #f2f5f8; }
        QFrame#statusCard, QFrame#activityPanel, QFrame#metricCard {
            background: #202630; border: 1px solid #303846; border-radius: 7px;
        }
        QFrame#metricCard { min-width: 170px; }
        QLabel#cardTitle { color: #aab3c0; font-size: 11px; }
        QTabWidget::pane {
            border: 0; border-top: 1px solid #303846; background: #161a20;
        }
        QTabBar::tab {
            background: #161a20; color: #929dab; padding: 11px 22px;
            border-bottom: 2px solid transparent;
        }
        QTabBar::tab:selected {
            color: #ffffff; border-bottom: 2px solid #4d83e6;
        }
        QTableWidget#notificationTable {
            background: #151a21; alternate-background-color: #1b212a;
            border: 1px solid #303846; border-radius: 7px;
            gridline-color: transparent; selection-background-color: #274b7f;
            selection-color: #ffffff;
        }
        QHeaderView::section {
            background: #202630; color: #aab3c0; border: 0;
            border-bottom: 1px solid #343d49; padding: 9px; font-weight: 600;
        }
        QTextEdit#notificationDetails {
            background: #202630; border: 1px solid #303846;
            border-radius: 7px; padding: 10px;
        }
        QPushButton {
            background: #2d6cdf; border: 0; border-radius: 5px;
            padding: 7px 11px; color: white;
        }
        QPushButton:hover { background: #3979ed; }
        QPushButton:disabled { background: #343b46; color: #747d89; }
        QLineEdit, QTextEdit, QComboBox {
            background: #101419; border: 1px solid #343d49;
            border-radius: 4px; padding: 6px; color: #e1e5ea;
        }
        QToolTip { background: #252b34; color: white; border: 1px solid #48515e; }
        """


def main() -> int:
    application = QApplication(sys.argv)
    application.setApplicationName("MRY Alert Control")
    application.setOrganizationName("MRY Jet Center")
    application.setQuitOnLastWindowClosed(True)
    window = ControlWindow()
    window.show()
    return application.exec()
