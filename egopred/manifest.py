"""Per-item processing state (pending -> done -> verified) backed by SQLite.

Every pipeline stage records what it has finished, so re-running a command skips
completed work and a crash only loses the item in flight.
"""

import json
import sqlite3
import time
from pathlib import Path

STATES = ("pending", "done", "verified", "failed")


class Manifest:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # several ingest workers may share one manifest: wait for the write lock
        self._db = sqlite3.connect(path, timeout=60)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS items ("
            " kind TEXT NOT NULL, key TEXT NOT NULL, state TEXT NOT NULL,"
            " info TEXT NOT NULL DEFAULT '{}', updated REAL NOT NULL,"
            " PRIMARY KEY (kind, key))"
        )
        self._db.commit()

    def get(self, kind: str, key: str) -> tuple[str, dict] | None:
        row = self._db.execute(
            "SELECT state, info FROM items WHERE kind = ? AND key = ?", (kind, key)
        ).fetchone()
        return None if row is None else (row[0], json.loads(row[1]))

    def state(self, kind: str, key: str) -> str | None:
        item = self.get(kind, key)
        return None if item is None else item[0]

    def set(self, kind: str, key: str, state: str, **info) -> None:
        """Set the state of an item, merging `info` into what was already stored."""
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        old = self.get(kind, key)
        merged = {**(old[1] if old else {}), **info}
        self._db.execute(
            "INSERT INTO items (kind, key, state, info, updated) VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT (kind, key) DO UPDATE SET"
            " state = excluded.state, info = excluded.info, updated = excluded.updated",
            (kind, key, state, json.dumps(merged), time.time()),
        )
        self._db.commit()

    def keys(self, kind: str, state: str | None = None) -> list[str]:
        if state is None:
            rows = self._db.execute("SELECT key FROM items WHERE kind = ? ORDER BY key", (kind,))
        else:
            rows = self._db.execute(
                "SELECT key FROM items WHERE kind = ? AND state = ? ORDER BY key", (kind, state)
            )
        return [r[0] for r in rows]

    def merge(self, other: Path) -> int:
        """Import the items of another manifest (e.g. from a remote run), keeping the most
        recently updated row when both have the same item. Returns rows taken from `other`."""
        before = self._db.total_changes
        self._db.execute("ATTACH DATABASE ? AS other", (str(other),))
        self._db.execute(
            "INSERT INTO items (kind, key, state, info, updated)"
            " SELECT kind, key, state, info, updated FROM other.items WHERE true"
            " ON CONFLICT (kind, key) DO UPDATE SET"
            " state = excluded.state, info = excluded.info, updated = excluded.updated"
            " WHERE excluded.updated > items.updated"
        )
        self._db.commit()
        self._db.execute("DETACH DATABASE other")
        return self._db.total_changes - before

    def summary(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for kind, state, n in self._db.execute(
            "SELECT kind, state, COUNT(*) FROM items GROUP BY kind, state ORDER BY kind"
        ):
            out.setdefault(kind, {})[state] = n
        return out

    def close(self) -> None:
        self._db.close()
