from __future__ import annotations

import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "extension"


def png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert data[12:16] == b"IHDR"
    return struct.unpack(">II", data[16:24])


def test_manifest_and_notification_icons_are_packaged_pngs() -> None:
    manifest = json.loads((EXTENSION / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 3
    assert "notifications" in manifest["permissions"]
    assert {"activeTab", "offscreen", "tabCapture"} <= set(manifest["permissions"])

    expected = {"16": 16, "32": 32, "48": 48, "128": 128}
    for key, size in expected.items():
        icon_path = EXTENSION / manifest["icons"][key]
        assert icon_path.is_file()
        assert png_dimensions(icon_path) == (size, size)

    background = (EXTENSION / "background.js").read_text(encoding="utf-8")
    assert 'chrome.runtime.getURL("icons/icon128.png")' in background
    assert 'iconUrl: "icon.svg"' not in background
    assert background.count("chrome.notifications.create(") == 1
    assert "chrome.runtime.lastError" in background
    assert "let connectPromise;" in background
    assert "socket !== currentSocket" in background
    assert "event.code === 4403" in background
    assert "connect(true)" in background

    for filename in ("offscreen.html", "offscreen.js", "audio-processor.js"):
        assert (EXTENSION / filename).is_file()
    offscreen = (EXTENSION / "offscreen.js").read_text(encoding="utf-8")
    assert '"/ws/audio?token="' in offscreen
    assert "MediaRecorder" not in offscreen
    assert "chromeMediaSourceId" in offscreen
