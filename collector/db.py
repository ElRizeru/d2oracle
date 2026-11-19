"""SQLite WAL state, discovery queues, and crash-safe match buffer."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from shared.constants import STEAM_ANONYMOUS_ACCOUNT_ID


class CollectorDB:
    """Persistent SQLite WAL queue and buffer for the match collector."""

    DISCOVERY_META_KEYS = (
        "opendota_public_cursor",
        "opendota_pro_newest_id",
        "opendota_pro_backfill_done",
        "opendota_pro_backfill_cursor",
        "leaderboard_cursors",
    )

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        if str(db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=10000;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute("PRAGMA journal_size_limit=16777216;")
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS match_queue (
                    match_id INTEGER PRIMARY KEY,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    priority INTEGER NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    reject_reason TEXT,
                    discovered_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS players_queue (
                    steam_account_id INTEGER PRIMARY KEY,
                    source TEXT NOT NULL,
                    last_fetched_at INTEGER,
                    last_match_id INTEGER,
                    priority INTEGER NOT NULL DEFAULT 0,
                    added_at INTEGER NOT NULL
                );

                DROP INDEX IF EXISTS idx_players_queue_fetch;

                CREATE INDEX IF NOT EXISTS idx_players_queue_unfetched
                    ON players_queue (priority DESC, added_at ASC)
                    WHERE last_fetched_at IS NULL;

                CREATE INDEX IF NOT EXISTS idx_players_queue_fetched
                    ON players_queue (last_fetched_at ASC, priority DESC, added_at ASC)
                    WHERE last_fetched_at IS NOT NULL;

                CREATE TABLE IF NOT EXISTS match_buffer (
                    match_id INTEGER PRIMARY KEY,
                    game_version_id INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_match_buffer_gv
                    ON match_buffer (game_version_id, created_at);
                """
            )
            cols = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(match_queue)").fetchall()
            }
            if "priority" not in cols:
                self._conn.execute(
                    "ALTER TABLE match_queue ADD COLUMN priority INTEGER NOT NULL DEFAULT 0;"
                )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_match_queue_status "
                "ON match_queue (status, priority DESC, attempts ASC, match_id DESC);"
            )
            idx_cols = [
                row["name"]
                for row in self._conn.execute("PRAGMA index_info(idx_match_queue_status)").fetchall()
            ]
            if idx_cols != ["status", "priority", "attempts", "match_id"]:
                self._conn.execute("DROP INDEX IF EXISTS idx_match_queue_status;")
                self._conn.execute(
                    "CREATE INDEX idx_match_queue_status "
                    "ON match_queue (status, priority DESC, attempts ASC, match_id DESC);"
                )

    def close(self) -> None:
        self._conn.close()

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def delete_meta(self, key: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM meta WHERE key = ?", (key,))

    def get_active_patch(self) -> dict[str, Any] | None:
        raw = self.get_meta("active_patch")
        return json.loads(raw) if raw else None

    def set_active_patch(
        self,
        gv_id: int,
        gv_name: str,
        as_of_datetime: int,
        *,
        min_start_datetime: int | None = None,
        valve_ahead: bool = False,
    ) -> None:
        payload: dict[str, Any] = {
            "id": int(gv_id),
            "name": str(gv_name),
            "asOfDateTime": int(as_of_datetime),
        }
        if min_start_datetime is not None:
            payload["minStartDateTime"] = int(min_start_datetime)
        if valve_ahead:
            payload["valveAhead"] = True
        self.set_meta("active_patch", json.dumps(payload))

    @staticmethod
    def _priority_for_source(source: str) -> int:
        return 10 if source in ("opendota_pro", "steam_live_league") else 0

    def enqueue_matches(self, matches: list[tuple[int, str]]) -> int:
        """Insert newly discovered match IDs ignoring duplicates. Returns count inserted."""
        if not matches:
            return 0
        now = int(time.time())
        rows = [
            (int(mid), str(src), self._priority_for_source(str(src)), now, now)
            for mid, src in matches
            if mid and int(mid) > 0
        ]
        if not rows:
            return 0
        high_prio = [(prio, src, mid, prio) for mid, src, prio, _, _ in rows if prio > 0]
        with self._conn:
            before = self._conn.total_changes
            self._conn.executemany(
                "INSERT OR IGNORE INTO match_queue "
                "(match_id, source, status, priority, discovered_at, updated_at) "
                "VALUES (?, ?, 'pending', ?, ?, ?)",
                rows,
            )
            inserted = self._conn.total_changes - before
            if high_prio:
                self._conn.executemany(
                    "UPDATE match_queue SET priority = ?, source = ?, updated_at = ? "
                    "WHERE match_id = ? AND status = 'pending' AND priority < ?",
                    [(prio, src, now, mid, max_p) for prio, src, mid, max_p in high_prio],
                )
            return inserted

    def pop_pending_matches(
        self, limit: int = 20, retry_cooldown_seconds: int = 0
    ) -> list[tuple[int, str, int]]:
        """Return pending (match_id, source, attempts) prioritized by priority, lowest attempts, newest match_id."""
        if retry_cooldown_seconds > 0:
            cutoff = int(time.time()) - int(retry_cooldown_seconds)
            rows = self._conn.execute(
                "SELECT match_id, source, attempts FROM match_queue "
                "WHERE status = 'pending' AND (attempts = 0 OR updated_at <= ?) "
                "ORDER BY priority DESC, attempts ASC, match_id DESC LIMIT ?",
                (cutoff, int(limit)),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT match_id, source, attempts FROM match_queue "
                "WHERE status = 'pending' "
                "ORDER BY priority DESC, attempts ASC, match_id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [(int(r["match_id"]), str(r["source"]), int(r["attempts"])) for r in rows]

    def count_pending_matches(self, retry_cooldown_seconds: int = 0) -> int:
        """Count currently eligible pending matches in queue."""
        if retry_cooldown_seconds > 0:
            cutoff = int(time.time()) - int(retry_cooldown_seconds)
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM match_queue "
                "WHERE status = 'pending' AND (attempts = 0 OR updated_at <= ?)",
                (cutoff,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM match_queue WHERE status = 'pending'"
            ).fetchone()
        return int(row["c"]) if row else 0

    def mark_match_collected(
        self, match_id: int, game_version_id: int, record: dict[str, Any]
    ) -> None:
        now = int(time.time())
        payload_json = json.dumps(record, separators=(",", ":"))
        with self._conn:
            self._conn.execute(
                "UPDATE match_queue SET status = 'collected', reject_reason = NULL, "
                "attempts = attempts + 1, updated_at = ? WHERE match_id = ?",
                (now, int(match_id)),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO match_buffer (match_id, game_version_id, payload_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (int(match_id), int(game_version_id), payload_json, now),
            )

    def mark_match_rejected(self, match_id: int, reason: str) -> None:
        now = int(time.time())
        with self._conn:
            self._conn.execute(
                "UPDATE match_queue SET status = 'rejected', reject_reason = ?, "
                "attempts = attempts + 1, updated_at = ? WHERE match_id = ?",
                (reason, now, int(match_id)),
            )

    def mark_match_retry(self, match_id: int, reason: str, max_attempts: int = 2) -> str:
        """Increment attempts; reject permanently once max_attempts is reached. Returns new status."""
        now = int(time.time())
        with self._conn:
            row = self._conn.execute(
                "SELECT attempts FROM match_queue WHERE match_id = ?", (int(match_id),)
            ).fetchone()
            attempts = (int(row["attempts"]) if row else 0) + 1
            new_status = "rejected" if attempts >= max_attempts else "pending"
            self._conn.execute(
                "UPDATE match_queue SET status = ?, reject_reason = ?, attempts = ?, updated_at = ? "
                "WHERE match_id = ?",
                (new_status, reason, attempts, now, int(match_id)),
            )
            return new_status

    def record_batch_results(
        self,
        *,
        game_version_id: int,
        collected: list[tuple[int, dict[str, Any]]],
        rejected: list[tuple[int, str]],
        retries: list[tuple[int, str, int]],
        bfs_player_ids: list[int],
    ) -> dict[int, str]:
        """Persist outcomes for a batch of validated matches in a single SQLite transaction."""
        now = int(time.time())
        clean_players = sorted(
            {
                int(sid)
                for sid in bfs_player_ids
                if sid and 0 < int(sid) != STEAM_ANONYMOUS_ACCOUNT_ID
            }
        )
        retry_statuses: dict[int, str] = {}

        with self._conn:
            if collected:
                coll_queue_rows = [(now, int(mid)) for mid, _ in collected]
                self._conn.executemany(
                    "UPDATE match_queue SET status = 'collected', reject_reason = NULL, "
                    "attempts = attempts + 1, updated_at = ? WHERE match_id = ?",
                    coll_queue_rows,
                )
                buf_rows = [
                    (
                        int(mid),
                        int(game_version_id),
                        json.dumps(record, separators=(",", ":")),
                        now,
                    )
                    for mid, record in collected
                ]
                self._conn.executemany(
                    "INSERT OR REPLACE INTO match_buffer "
                    "(match_id, game_version_id, payload_json, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    buf_rows,
                )

            if clean_players:
                player_rows = [(sid, "match_bfs", 10, now) for sid in clean_players]
                self._conn.executemany(
                    "INSERT INTO players_queue (steam_account_id, source, priority, added_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(steam_account_id) DO UPDATE SET "
                    "last_fetched_at = CASE "
                    "WHEN excluded.priority >= 10 AND players_queue.priority < 10 THEN NULL "
                    "ELSE players_queue.last_fetched_at END, "
                    "priority = MAX(players_queue.priority, excluded.priority)",
                    player_rows,
                )

            if rejected:
                rej_rows = [
                    (str(reason), now, int(mid)) for mid, reason in rejected
                ]
                self._conn.executemany(
                    "UPDATE match_queue SET status = 'rejected', reject_reason = ?, "
                    "attempts = attempts + 1, updated_at = ? WHERE match_id = ?",
                    rej_rows,
                )

            if retries:
                retry_ids = [int(mid) for mid, _reason, _max_att in retries]
                placeholders = ",".join("?" for _ in retry_ids)
                existing_rows = self._conn.execute(
                    f"SELECT match_id, attempts FROM match_queue WHERE match_id IN ({placeholders})",
                    retry_ids,
                ).fetchall()
                attempts_map = {
                    int(r["match_id"]): int(r["attempts"]) for r in existing_rows
                }
                retry_updates: list[tuple[str, str, int, int, int]] = []
                for mid, reason, max_attempts in retries:
                    mid_int = int(mid)
                    next_attempts = attempts_map.get(mid_int, 0) + 1
                    new_status = "rejected" if next_attempts >= max_attempts else "pending"
                    retry_statuses[mid_int] = new_status
                    retry_updates.append((new_status, str(reason), next_attempts, now, mid_int))
                self._conn.executemany(
                    "UPDATE match_queue SET status = ?, reject_reason = ?, attempts = ?, updated_at = ? "
                    "WHERE match_id = ?",
                    retry_updates,
                )

        return retry_statuses

    def has_players(self) -> bool:
        """O(1) existence check avoiding full table/index scans on players_queue."""
        row = self._conn.execute("SELECT 1 FROM players_queue LIMIT 1").fetchone()
        return row is not None

    def enqueue_players(
        self, steam_account_ids: list[int], source: str, priority: int = 0
    ) -> int:
        """Insert Immortal player seeds for BFS without overwriting fetch cursors."""
        clean_ids = sorted(
            {
                int(sid)
                for sid in steam_account_ids
                if sid and 0 < int(sid) != STEAM_ANONYMOUS_ACCOUNT_ID
            }
        )
        if not clean_ids:
            return 0
        now = int(time.time())
        rows = [(sid, source, int(priority), now) for sid in clean_ids]
        with self._conn:
            before = self._conn.total_changes
            self._conn.executemany(
                "INSERT INTO players_queue (steam_account_id, source, priority, added_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(steam_account_id) DO UPDATE SET "
                "last_fetched_at = CASE "
                "WHEN excluded.priority >= 10 AND players_queue.priority < 10 THEN NULL "
                "ELSE players_queue.last_fetched_at END, "
                "priority = MAX(players_queue.priority, excluded.priority)",
                rows,
            )
            return self._conn.total_changes - before

    def pop_players_for_bfs(
        self, limit: int = 10, cooldown_seconds: int = 14400
    ) -> list[tuple[int, int | None]]:
        """Select players eligible for BFS polling (active-patch priority >= 10 before fallback < 10)."""
        lim = max(0, int(limit))
        if lim == 0:
            return []

        cutoff = int(time.time()) - int(cooldown_seconds)
        rows = list(
            self._conn.execute(
                "SELECT steam_account_id, last_match_id FROM players_queue "
                "WHERE last_fetched_at IS NULL AND priority >= 10 "
                "ORDER BY priority DESC, added_at ASC "
                "LIMIT ?",
                (lim,),
            ).fetchall()
        )
        if len(rows) < lim:
            rows.extend(
                self._conn.execute(
                    "SELECT steam_account_id, last_match_id FROM players_queue "
                    "WHERE last_fetched_at IS NOT NULL AND last_fetched_at <= ? AND priority >= 10 "
                    "ORDER BY last_fetched_at ASC, priority DESC, added_at ASC "
                    "LIMIT ?",
                    (cutoff, lim - len(rows)),
                ).fetchall()
            )
        if len(rows) < lim:
            rows.extend(
                self._conn.execute(
                    "SELECT steam_account_id, last_match_id FROM players_queue "
                    "WHERE last_fetched_at IS NULL AND priority < 10 "
                    "ORDER BY priority DESC, added_at ASC "
                    "LIMIT ?",
                    (lim - len(rows),),
                ).fetchall()
            )
        if len(rows) < lim:
            rows.extend(
                self._conn.execute(
                    "SELECT steam_account_id, last_match_id FROM players_queue "
                    "WHERE last_fetched_at IS NOT NULL AND last_fetched_at <= ? AND priority < 10 "
                    "ORDER BY last_fetched_at ASC, priority DESC, added_at ASC "
                    "LIMIT ?",
                    (cutoff, lim - len(rows)),
                ).fetchall()
            )
        return [
            (int(r["steam_account_id"]), int(r["last_match_id"]) if r["last_match_id"] is not None else None)
            for r in rows
        ]

    def mark_player_fetched(self, steam_account_id: int, last_match_id: int | None) -> None:
        now = int(time.time())
        with self._conn:
            self._conn.execute(
                "UPDATE players_queue SET last_fetched_at = ?, "
                "last_match_id = COALESCE(?, last_match_id) "
                "WHERE steam_account_id = ?",
                (now, int(last_match_id) if last_match_id is not None else None, int(steam_account_id)),
            )

    def mark_players_fetched_batch(
        self, updates: list[tuple[int, int | None]]
    ) -> None:
        """Update BFS polling timestamps and watermarks for a batch of players in one transaction."""
        if not updates:
            return
        now = int(time.time())
        rows = [
            (now, int(last_mid) if last_mid is not None else None, int(sid))
            for sid, last_mid in updates
        ]
        with self._conn:
            self._conn.executemany(
                "UPDATE players_queue SET last_fetched_at = ?, "
                "last_match_id = COALESCE(?, last_match_id) "
                "WHERE steam_account_id = ?",
                rows,
            )

    def buffered_game_versions(self) -> list[int]:
        rows = self._conn.execute(
            "SELECT DISTINCT game_version_id FROM match_buffer ORDER BY game_version_id ASC"
        ).fetchall()
        return [int(r["game_version_id"]) for r in rows]

    def count_buffered_matches(self, game_version_id: int | None = None) -> int:
        if game_version_id is None:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM match_buffer").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM match_buffer WHERE game_version_id = ?",
                (int(game_version_id),),
            ).fetchone()
        return int(row["c"]) if row else 0

    def fetch_buffered_matches(
        self, game_version_id: int, limit: int | None = None
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT match_id, payload_json FROM match_buffer "
            "WHERE game_version_id = ? ORDER BY created_at ASC, match_id ASC"
        )
        params: tuple[Any, ...]
        if limit is not None:
            sql += " LIMIT ?"
            params = (int(game_version_id), int(limit))
        else:
            params = (int(game_version_id),)
        rows = self._conn.execute(sql, params).fetchall()
        return [json.loads(r["payload_json"]) for r in rows]

    def delete_buffered_matches(self, match_ids: list[int]) -> None:
        if not match_ids:
            return
        placeholders = ",".join("?" for _ in match_ids)
        with self._conn:
            self._conn.execute(
                f"DELETE FROM match_buffer WHERE match_id IN ({placeholders})",
                [int(mid) for mid in match_ids],
            )
        self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")

    def reset_for_new_patch(self) -> None:
        """Clear match queue and reset player & discovery cursors when switching patches."""
        with self._conn:
            self._conn.execute("DELETE FROM match_queue")
            self._conn.execute(
                "UPDATE players_queue SET last_fetched_at = NULL, last_match_id = NULL, "
                "priority = MIN(priority, 1)"
            )
            placeholders = ",".join("?" for _ in self.DISCOVERY_META_KEYS)
            self._conn.execute(
                f"DELETE FROM meta WHERE key IN ({placeholders})",
                self.DISCOVERY_META_KEYS,
            )

    def stats(self) -> dict[str, int]:
        out: dict[str, int] = {
            "pending": 0,
            "collected": 0,
            "rejected": 0,
            "players_total": 0,
            "players_unfetched": 0,
            "buffer_total": self.count_buffered_matches(),
        }
        for row in self._conn.execute(
            "SELECT status, COUNT(*) AS c FROM match_queue GROUP BY status"
        ).fetchall():
            out[str(row["status"])] = int(row["c"])
        p_row = self._conn.execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN last_fetched_at IS NULL THEN 1 ELSE 0 END) AS unfetched "
            "FROM players_queue"
        ).fetchone()
        if p_row:
            out["players_total"] = int(p_row["total"] or 0)
            out["players_unfetched"] = int(p_row["unfetched"] or 0)
        return out
