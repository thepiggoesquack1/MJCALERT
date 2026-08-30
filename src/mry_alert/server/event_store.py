from __future__ import annotations

from collections import deque
from pathlib import Path

from pydantic import ValidationError

from mry_alert.models import AlertEvent, NotificationRecord


class EventStore:
    def __init__(self, path: Path, maximum: int = 100) -> None:
        self.path = path
        self._events: deque[AlertEvent] = deque(maxlen=maximum)

    def append(self, event: AlertEvent) -> None:
        self._events.appendleft(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

    def recent(self) -> list[AlertEvent]:
        return list(self._events)


class NotificationHistoryStore:
    def __init__(self, path: Path, maximum: int = 250) -> None:
        self.path = path
        self.maximum = maximum
        self._records: deque[NotificationRecord] = deque(maxlen=maximum)
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        loaded: list[NotificationRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                loaded.append(NotificationRecord.model_validate_json(line))
            except ValidationError:
                continue
        self._records.extend(reversed(loaded[-self.maximum :]))

    def append(self, record: NotificationRecord) -> None:
        self._records.appendleft(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    def recent(self) -> list[NotificationRecord]:
        return list(self._records)
