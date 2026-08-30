from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QProcess, QTimer
from PySide6.QtWidgets import QApplication

from mry_alert.control.app import ControlWindow


def main() -> int:
    application = QApplication(sys.argv)
    window = ControlWindow()
    state = {"command_started": False, "test_sent": False, "stopping": False}

    def fail(message: str) -> None:
        print(f"CONTROL SMOKE FAILED: {message}", file=sys.stderr)
        if window.backend_owned:
            window.stop_server()
            window.backend_process.waitForFinished(5000)
        application.exit(1)

    def inspect() -> None:
        if state["stopping"]:
            if window.backend_process.state() == QProcess.ProcessState.NotRunning:
                print("CONTROL SMOKE PASSED")
                application.exit(0)
            return
        if not window.health_ok:
            return
        if not state["command_started"]:
            state["command_started"] = True
            window.command_input.setText(
                "simulate --fixture tests/fixtures/ambiguous.json --config config.yaml"
            )
            window.run_manual_command()
            return
        if window.command_process.state() != QProcess.ProcessState.NotRunning:
            return
        if not state["test_sent"]:
            state["test_sent"] = True
            window.test_alert()
            QTimer.singleShot(750, stop)

    def stop() -> None:
        state["stopping"] = True
        window.stop_server()

    QTimer.singleShot(100, window.start_server)
    timer = QTimer()
    timer.timeout.connect(inspect)
    timer.start(200)
    QTimer.singleShot(25000, lambda: fail("timed out"))
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
