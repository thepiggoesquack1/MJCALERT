from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from mry_alert.audio_classifier.dataset import (
    ReviewedLabels,
    load_metadata,
    review_clip,
    scan_clips,
)
from mry_alert.audio_classifier.models import DestinationLabel, IntentLabel


class ReviewDialog(QDialog):
    def __init__(self, dataset: Path, parent: object | None = None) -> None:
        super().__init__(parent)  # type: ignore[arg-type]
        self.dataset = dataset
        self.current_metadata: Path | None = None
        self.setWindowTitle("ATC Audio Review Queue")
        self.resize(920, 620)
        root = QHBoxLayout(self)
        self.clips = QListWidget()
        self.clips.currentTextChanged.connect(self._load_selected)
        root.addWidget(self.clips, 1)
        detail = QVBoxLayout()
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        detail.addWidget(self.summary, 2)
        form = QFormLayout()
        self.destination = QComboBox()
        self.destination.addItems([item.value for item in DestinationLabel])
        self.intent = QComboBox()
        self.intent.addItems([item.value for item in IntentLabel])
        self.correction = QCheckBox("Correction / destination change")
        self.unintelligible = QCheckBox("Unintelligible")
        self.hard_negative = QCheckBox("Hard negative")
        self.callsign = QLineEdit()
        form.addRow("Destination", self.destination)
        form.addRow("Intent", self.intent)
        form.addRow("", self.correction)
        form.addRow("", self.unintelligible)
        form.addRow("", self.hard_negative)
        form.addRow("Callsign / registration", self.callsign)
        detail.addLayout(form)
        buttons = QHBoxLayout()
        for label, callback in (
            ("Play Clip", self.play_clip),
            ("Save Review", self.save_review),
            ("Reject", self.reject_clip),
            ("Close", self.accept),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        detail.addLayout(buttons)
        root.addLayout(detail, 3)
        self.refresh()

    def refresh(self) -> None:
        self.clips.clear()
        self.current_metadata = None
        pending = self.dataset / "pending"
        scan = scan_clips(pending)
        for path in scan.valid_metadata:
            self.clips.addItem(path.name)
        if self.clips.count() == 0:
            message = "No valid clips are waiting for review."
        else:
            message = f"{self.clips.count()} valid clip(s) waiting for review."
        if scan.issues:
            details = "\n".join(
                f"- {issue.path.name}: {issue.code} — {issue.detail}" for issue in scan.issues
            )
            message += f"\n\nSkipped {len(scan.issues)} invalid item(s):\n{details}"
        self.summary.setPlainText(message)

    def _load_selected(self, name: str) -> None:
        if not name:
            self.current_metadata = None
            return
        self.current_metadata = self.dataset / "pending" / name
        try:
            metadata = load_metadata(self.current_metadata)
        except (OSError, ValueError) as exc:
            self.summary.setPlainText(f"Unable to load metadata: {exc}")
            return
        self.summary.setPlainText(
            f"Whisper transcript:\n{metadata.original_transcript or '(none)'}\n\n"
            f"Normalized:\n{metadata.normalized_transcript or '(none)'}\n\n"
            f"Detection: {metadata.detection_decision or '(none)'}\n"
            f"Speaker role: {metadata.speaker_role} "
            f"({metadata.speaker_role_confidence:.0%})\n"
            f"Destination: {metadata.destination_candidate or '(none)'}\n"
            f"Intent: {metadata.intent_category or '(none)'}\n\n"
            f"Classifier status: {metadata.classifier_status}\n"
            f"Classifier:\n{metadata.classifier_output.model_dump_json(indent=2)}\n\n"
            f"ADS-B candidates:\n{metadata.adsb_candidates or '(none)'}"
        )
        self.destination.setCurrentText(
            metadata.destination_candidate or metadata.classifier_output.destination.value
        )
        self.intent.setCurrentText(
            metadata.intent_category or metadata.classifier_output.intent.value
        )
        self.correction.setChecked(metadata.classifier_output.correction)

    def play_clip(self) -> None:
        if not self.current_metadata:
            return
        scan = scan_clips(self.current_metadata.parent)
        if self.current_metadata not in scan.valid_metadata:
            issue = next(
                (item for item in scan.issues if item.path == self.current_metadata),
                None,
            )
            QMessageBox.warning(
                self,
                "Invalid clip",
                issue.detail if issue else "This clip failed validation and was skipped.",
            )
            self.refresh()
            return
        metadata = load_metadata(self.current_metadata)
        if not metadata.wav_file:
            QMessageBox.information(self, "Audio", "This item has no saved WAV clip.")
            return
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self.current_metadata.parent / metadata.wav_file))
        )

    def _labels(self) -> ReviewedLabels:
        return ReviewedLabels(
            destination=self.destination.currentText(),
            intent=self.intent.currentText(),
            correction=self.correction.isChecked(),
            callsign_or_registration=self.callsign.text().strip() or None,
            unintelligible=self.unintelligible.isChecked(),
        )

    def save_review(self) -> None:
        if not self.current_metadata:
            return
        review_clip(
            self.current_metadata,
            self._labels(),
            hard_negative=self.hard_negative.isChecked(),
        )
        self.refresh()

    def reject_clip(self) -> None:
        if not self.current_metadata:
            return
        review_clip(self.current_metadata, self._labels(), reject=True)
        self.refresh()
