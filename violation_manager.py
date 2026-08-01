"""Tracks per-user violations and decides when a user should be temporarily
blocked from sending messages. JSON-file-backed, which is fine for a demo
or single-process app -- swap `_load`/`_save` for a real database (e.g.
SQLite/Postgres/Redis) before deploying this for real.
"""
import json
import time
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent
VIOLATIONS_PATH = BASE_DIR / "models" / "violations.json"

BLOCK_THRESHOLD = 3            # violations within the window before a block
BLOCK_WINDOW_SECONDS = 3600    # violations older than this are forgotten
BLOCK_DURATION_SECONDS = 600   # how long a user stays blocked once triggered


class ViolationManager:
    def __init__(
        self,
        path: Path = VIOLATIONS_PATH,
        threshold: int = BLOCK_THRESHOLD,
        window_seconds: int = BLOCK_WINDOW_SECONDS,
        block_duration_seconds: int = BLOCK_DURATION_SECONDS,
    ):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.window_seconds = window_seconds
        self.block_duration_seconds = block_duration_seconds
        self._data = self._load()

    def _load(self) -> Dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2))

    def _user_record(self, user_id: str) -> Dict:
        return self._data.setdefault(user_id, {"violations": [], "blocked_until": 0})

    def is_blocked(self, user_id: str) -> bool:
        record = self._user_record(user_id)
        return time.time() < record.get("blocked_until", 0)

    def seconds_until_unblocked(self, user_id: str) -> int:
        record = self._user_record(user_id)
        remaining = record.get("blocked_until", 0) - time.time()
        return max(0, int(remaining))

    def register_violation(self, user_id: str, severity_label: str) -> Dict:
        """Record a new violation, pruning stale ones, and decide whether to
        trigger a temporary block."""
        record = self._user_record(user_id)
        now = time.time()
        record["violations"] = [
            t for t in record["violations"] if now - t < self.window_seconds
        ]
        record["violations"].append(now)

        blocked = False
        if severity_label == "severe" or len(record["violations"]) >= self.threshold:
            record["blocked_until"] = now + self.block_duration_seconds
            blocked = True

        self._save()
        return {
            "violation_count": len(record["violations"]),
            "blocked": blocked,
            "blocked_until": record.get("blocked_until", 0),
        }

    def reset_user(self, user_id: str) -> None:
        self._data.pop(user_id, None)
        self._save()
