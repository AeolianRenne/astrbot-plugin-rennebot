"""SQLite persistence for the game registry and future cache-backed commands."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc


@dataclass(frozen=True)
class GameRecord:
    """A game-ID record scoped to one QQ group."""

    group_id: str
    user_id: str
    display_name: str
    game_name: str
    game_id: str
    updated_at: str


@dataclass(frozen=True)
class PrivateAIConversation:
    """Persisted private AI session data for one QQ platform user."""

    active: bool
    summary: str
    messages: list[dict[str, str]]


class PluginDatabase:
    """Store plugin data in a self-contained SQLite database."""

    def __init__(self, path: Path) -> None:
        """Create a database facade.

        Args:
            path: Database file path in AstrBot's persistent plugin data directory.
        """
        self.path = path

    def initialize(self) -> None:
        """Create the current schema when it does not yet exist."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS game_ids (
                    group_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    game_name TEXT NOT NULL COLLATE NOCASE,
                    game_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (group_id, user_id, game_name)
                );
                CREATE INDEX IF NOT EXISTS idx_game_ids_group_game
                    ON game_ids (group_id, game_name, display_name);
                CREATE TABLE IF NOT EXISTS cache_entries (
                    namespace TEXT NOT NULL,
                    scope_type TEXT NOT NULL CHECK (scope_type IN ('global', 'group', 'user')),
                    scope_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    expires_at TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (namespace, scope_type, scope_id, key)
                );
                CREATE INDEX IF NOT EXISTS idx_cache_expiry ON cache_entries (expires_at);
                CREATE TABLE IF NOT EXISTS plugin_settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', '1')"
            )

    def upsert_game_id(
        self,
        group_id: str,
        user_id: str,
        display_name: str,
        game_name: str,
        game_id: str,
    ) -> None:
        """Insert or update a member's ID for a game in a group.

        Args:
            group_id: QQ group platform ID.
            user_id: QQ sender platform ID.
            display_name: Current sender display name.
            game_name: Game name supplied by the sender.
            game_id: Validated numeric game ID.
        """
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO game_ids(group_id, user_id, display_name, game_name, game_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(group_id, user_id, game_name) DO UPDATE SET
                    display_name = excluded.display_name,
                    game_id = excluded.game_id,
                    updated_at = excluded.updated_at
                """,
                (group_id, user_id, display_name, game_name, game_id, _now()),
            )

    def list_game_ids(self, group_id: str, game_name: str) -> list[GameRecord]:
        """List IDs registered for a game in one group.

        Args:
            group_id: QQ group platform ID.
            game_name: Game name to look up case-insensitively.

        Returns:
            Records sorted by display name and sender ID.
        """
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT group_id, user_id, display_name, game_name, game_id, updated_at
                FROM game_ids
                WHERE group_id = ? AND game_name = ? COLLATE NOCASE
                ORDER BY display_name COLLATE NOCASE, user_id
                """,
                (group_id, game_name),
            ).fetchall()
        return [GameRecord(**dict(row)) for row in rows]

    def delete_game_id(self, group_id: str, user_id: str, game_name: str) -> bool:
        """Delete one member's game ID in a group.

        Args:
            group_id: QQ group platform ID.
            user_id: QQ sender platform ID of the record to remove.
            game_name: Game name of the record to remove.

        Returns:
            Whether a record was deleted.
        """
        with self._connection() as connection:
            result = connection.execute(
                "DELETE FROM game_ids WHERE group_id = ? AND user_id = ? "
                "AND game_name = ? COLLATE NOCASE",
                (group_id, user_id, game_name),
            )
        return result.rowcount > 0

    def set_cache(
        self,
        namespace: str,
        scope_type: str,
        scope_id: str,
        key: str,
        value: Any,
        expires_at: datetime | None = None,
    ) -> None:
        """Persist a JSON-serializable value, optionally with an expiry time.

        Args:
            namespace: Feature-owned cache namespace.
            scope_type: One of global, group, or user.
            scope_id: Scope identifier; use an empty string for global data.
            key: Entry key inside the namespace and scope.
            value: JSON-serializable value.
            expires_at: UTC-aware expiry timestamp, if the entry should expire.

        Raises:
            ValueError: If scope_type is unsupported.
        """
        if scope_type not in {"global", "group", "user"}:
            raise ValueError("scope_type must be global, group, or user")
        expiry = expires_at.astimezone(UTC).isoformat() if expires_at else None
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO cache_entries(namespace, scope_type, scope_id, key, value_json, expires_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, scope_type, scope_id, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                """,
                (
                    namespace,
                    scope_type,
                    scope_id,
                    key,
                    json.dumps(value),
                    expiry,
                    _now(),
                ),
            )

    def get_cache(
        self, namespace: str, scope_type: str, scope_id: str, key: str
    ) -> Any | None:
        """Read a cache value and remove it when it has expired.

        Args:
            namespace: Feature-owned cache namespace.
            scope_type: One of global, group, or user.
            scope_id: Scope identifier.
            key: Entry key inside the namespace and scope.

        Returns:
            The deserialized value, or None when missing or expired.
        """
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT value_json, expires_at FROM cache_entries
                WHERE namespace = ? AND scope_type = ? AND scope_id = ? AND key = ?
                """,
                (namespace, scope_type, scope_id, key),
            ).fetchone()
            if not row:
                return None
            if row["expires_at"] and datetime.fromisoformat(
                row["expires_at"]
            ) <= datetime.now(UTC):
                connection.execute(
                    "DELETE FROM cache_entries WHERE namespace = ? AND scope_type = ? "
                    "AND scope_id = ? AND key = ?",
                    (namespace, scope_type, scope_id, key),
                )
                return None
        return json.loads(row["value_json"])

    def get_private_ai_conversation(self, user_id: str) -> PrivateAIConversation:
        """Read one user's private AI session with defensive JSON validation.

        Args:
            user_id: QQ platform user ID that owns the session.

        Returns:
            The normalized session state, or an inactive empty state when absent.
        """
        value = self.get_cache("private_ai", "user", user_id, "conversation")
        if not isinstance(value, dict):
            return PrivateAIConversation(False, "", [])
        messages: list[dict[str, str]] = []
        raw_messages = value.get("messages", [])
        if isinstance(raw_messages, list):
            for item in raw_messages:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                content = item.get("content")
                if role in {"user", "assistant"} and isinstance(content, str):
                    messages.append({"role": role, "content": content})
        summary = value.get("summary", "")
        return PrivateAIConversation(
            active=bool(value.get("active", False)),
            summary=summary if isinstance(summary, str) else "",
            messages=messages,
        )

    def set_private_ai_conversation(
        self,
        user_id: str,
        active: bool,
        summary: str,
        messages: list[dict[str, str]],
    ) -> None:
        """Persist one user's private AI session.

        Args:
            user_id: QQ platform user ID that owns the session.
            active: Whether ordinary private messages should call AI.
            summary: Compact memory of older conversation turns.
            messages: Recent user and assistant turns to send verbatim.
        """
        self.set_cache(
            "private_ai",
            "user",
            user_id,
            "conversation",
            {"active": active, "summary": summary, "messages": messages},
        )

    def set_setting(self, key: str, value: Any) -> None:
        """Persist a plugin-wide JSON setting.

        Args:
            key: Unique setting name.
            value: JSON-serializable setting value.
        """
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO plugin_settings(key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (key, json.dumps(value), _now()),
            )

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Read a plugin-wide JSON setting.

        Args:
            key: Unique setting name.
            default: Value to return when the setting is absent.

        Returns:
            Stored deserialized value, or default when the setting is absent.
        """
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value_json FROM plugin_settings WHERE key = ?",
                (key,),
            ).fetchone()
        return json.loads(row["value_json"]) if row else default

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a transaction-safe SQLite connection.

        Yields:
            A SQLite connection configured for concurrent readers.
        """
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _now() -> str:
    """Return the current UTC timestamp in ISO 8601 format.

    Returns:
        Current UTC timestamp.
    """
    return datetime.now(UTC).isoformat()
