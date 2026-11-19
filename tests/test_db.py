"""Unit tests for SQLite WAL queue and buffer persistence."""

from __future__ import annotations

from pathlib import Path

from collector.db import CollectorDB


def test_db_wal_mode_and_dedup_enqueue(tmp_path: Path) -> None:
    db = CollectorDB(tmp_path / "test.sqlite3")
    try:
        mode = db._conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert str(mode).lower() == "wal"

        inserted_1 = db.enqueue_matches([(101, "bfs"), (102, "opendota_pro"), (101, "opendota_public")])
        assert inserted_1 == 2

        inserted_2 = db.enqueue_matches([(102, "bfs"), (103, "bfs")])
        assert inserted_2 == 1

        pending = db.pop_pending_matches(limit=10)
        # opendota_pro has higher priority than bfs, ties break by newest match_id DESC
        assert [m[0] for m in pending] == [102, 103, 101]

        # Existing pending bfs match 101 promoted when also seen as opendota_pro
        db.enqueue_matches([(101, "opendota_pro")])
        pending_after = db.pop_pending_matches(limit=10)
        assert [(m[0], m[1]) for m in pending_after] == [
            (102, "opendota_pro"),
            (101, "opendota_pro"),
            (103, "bfs"),
        ]
    finally:
        db.close()


def test_db_retry_transitions_to_rejected_and_buffer_lifecycle(tmp_path: Path) -> None:
    db = CollectorDB(tmp_path / "test.sqlite3")
    try:
        db.enqueue_matches([(201, "bfs"), (202, "bfs")])
        assert db.mark_match_retry(201, "unparsed", max_attempts=2) == "pending"
        assert db.mark_match_retry(201, "unparsed", max_attempts=2) == "rejected"

        db.mark_match_collected(202, 190, {"match_id": 202, "start_datetime": 1000})
        assert db.count_buffered_matches(190) == 1
        rows = db.fetch_buffered_matches(190)
        assert [r["match_id"] for r in rows] == [202]
        db.delete_buffered_matches([202])
        assert db.count_buffered_matches(190) == 0
    finally:
        db.close()


def test_db_players_queue_priority_and_patch_reset(tmp_path: Path) -> None:
    db = CollectorDB(tmp_path / "test.sqlite3")
    try:
        db.enqueue_players([10, 20], source="leaderboard_tail", priority=0)
        db.enqueue_players([30], source="match_bfs", priority=10)

        popped = db.pop_players_for_bfs(limit=10, cooldown_seconds=1800)
        assert [p[0] for p in popped] == [30, 10, 20]

        db.mark_player_fetched(30, 9001)
        popped_after = db.pop_players_for_bfs(limit=10, cooldown_seconds=1800)
        assert [p[0] for p in popped_after] == [10, 20]

        db.enqueue_matches([(500, "bfs")])
        db.set_meta("opendota_public_cursor", "888")
        db.reset_for_new_patch()

        assert db.pop_pending_matches(limit=10) == []
        assert db.get_meta("opendota_public_cursor") is None
        # Player seeds remain known, with reset cursors so BFS starts fresh on new patch
        popped_reset = db.pop_players_for_bfs(limit=10, cooldown_seconds=1800)
        assert [p[0] for p in popped_reset] == [30, 10, 20]
        assert all(p[1] is None for p in popped_reset)

        # Newly discovered active-patch players (priority >= 10) pop ahead of demoted old-patch seeds
        db.enqueue_players([40], source="match_bfs", priority=10)
        popped_new_patch = db.pop_players_for_bfs(limit=10, cooldown_seconds=1800)
        assert [p[0] for p in popped_new_patch] == [40, 30, 10, 20]
    finally:
        db.close()


def test_pending_match_retry_cooldown_and_count(tmp_path: Path) -> None:
    db = CollectorDB(tmp_path / "cooldown.sqlite3")
    try:
        db.enqueue_matches([(301, "bfs"), (302, "bfs")])
        assert db.count_pending_matches(retry_cooldown_seconds=1800) == 2

        assert db.mark_match_retry(301, "missing_playback_data", max_attempts=5) == "pending"
        # 301 is now in cooldown so only fresh 302 is returned/counted
        assert db.count_pending_matches(retry_cooldown_seconds=1800) == 1
        assert [m[0] for m in db.pop_pending_matches(limit=10, retry_cooldown_seconds=1800)] == [302]

        # Once cooldown elapses, 301 becomes eligible again
        db._conn.execute("UPDATE match_queue SET updated_at = updated_at - 1900 WHERE match_id = 301")
        db._conn.commit()
        assert db.count_pending_matches(retry_cooldown_seconds=1800) == 2
        assert [m[0] for m in db.pop_pending_matches(limit=10, retry_cooldown_seconds=1800)] == [302, 301]
    finally:
        db.close()


def test_db_partial_indexes_batch_operations_and_wal_cap(tmp_path: Path) -> None:
    db = CollectorDB(tmp_path / "batch_ops.sqlite3")
    try:
        index_names = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='players_queue'"
            ).fetchall()
        }
        assert "idx_players_queue_fetch" not in index_names
        assert "idx_players_queue_unfetched" in index_names
        assert "idx_players_queue_fetched" in index_names

        journal_limit = db._conn.execute("PRAGMA journal_size_limit;").fetchone()[0]
        assert int(journal_limit) == 16777216

        assert not db.has_players()
        db.enqueue_players([101, 102, 103], source="match_bfs", priority=10)
        assert db.has_players()

        plan_unfetched = " ".join(
            str(r[3])
            for r in db._conn.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT steam_account_id FROM players_queue
                WHERE last_fetched_at IS NULL
                ORDER BY priority DESC, added_at ASC
                LIMIT 10
                """
            ).fetchall()
        )
        plan_fetched = " ".join(
            str(r[3])
            for r in db._conn.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT steam_account_id FROM players_queue
                WHERE last_fetched_at IS NOT NULL AND last_fetched_at <= 1000
                ORDER BY last_fetched_at ASC, priority DESC
                LIMIT 10
                """
            ).fetchall()
        )
        assert "TEMP B-TREE" not in plan_unfetched.upper()
        assert "TEMP B-TREE" not in plan_fetched.upper()

        db.mark_players_fetched_batch([(101, 5000), (102, 5001)])
        popped = db.pop_players_for_bfs(limit=5, cooldown_seconds=14400)
        assert popped == [(103, None)]

        db._conn.execute("UPDATE players_queue SET last_fetched_at = last_fetched_at - 15000 WHERE steam_account_id = 101")
        db._conn.commit()
        assert db.pop_players_for_bfs(limit=5, cooldown_seconds=14400) == [(103, None), (101, 5000)]

        db.enqueue_matches([(201, "bfs"), (202, "bfs"), (203, "bfs"), (204, "opendota_public")])
        outcomes = db.record_batch_results(
            game_version_id=190,
            collected=[(201, {"match_id": 201, "game_version_id": 190})],
            rejected=[(202, "wrong_patch")],
            retries=[(203, "missing_playback_data", 2), (204, "unparsed", 1)],
            bfs_player_ids=[9001, 9002],
        )
        assert outcomes == {203: "pending", 204: "rejected"}
        queue_statuses = dict(
            db._conn.execute("SELECT match_id, status FROM match_queue WHERE match_id IN (201, 202, 203, 204)").fetchall()
        )
        assert queue_statuses == {201: "collected", 202: "rejected", 203: "pending", 204: "rejected"}
        enqueued_bfs = {
            r[0]
            for r in db._conn.execute(
                "SELECT steam_account_id FROM players_queue WHERE steam_account_id IN (9001, 9002)"
            ).fetchall()
        }
        assert enqueued_bfs == {9001, 9002}
        assert db.count_buffered_matches(190) == 1
        db.delete_buffered_matches([201])
        assert db.count_buffered_matches(190) == 0
    finally:
        db.close()
