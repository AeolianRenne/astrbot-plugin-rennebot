import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from qq_game_registry.database import PluginDatabase


def test_game_ids_are_isolated_by_group_and_upserted(tmp_path) -> None:
    database = PluginDatabase(tmp_path / "rennebot.sqlite3")
    database.initialize()

    database.upsert_game_id("group-a", "user-1", "Alice", "原神", "100")
    database.upsert_game_id("group-a", "user-1", "Alice New", "原神", "200")
    database.upsert_game_id("group-b", "user-1", "Alice", "原神", "300")

    records = database.list_game_ids("group-a", "原神")
    assert [(record.display_name, record.game_id) for record in records] == [
        ("Alice New", "200")
    ]
    assert database.list_game_ids("group-b", "原神")[0].game_id == "300"


def test_delete_game_id_is_scoped_to_user_and_group(tmp_path) -> None:
    database = PluginDatabase(tmp_path / "rennebot.sqlite3")
    database.initialize()
    database.upsert_game_id("group-a", "user-1", "Alice", "原神", "100")

    assert not database.delete_game_id("group-a", "user-2", "原神")
    assert database.delete_game_id("group-a", "user-1", "原神")
    assert database.list_game_ids("group-a", "原神") == []


def test_cache_persists_and_expires(tmp_path) -> None:
    database = PluginDatabase(tmp_path / "rennebot.sqlite3")
    database.initialize()
    database.set_cache("example", "group", "group-a", "key", {"enabled": True})
    database.set_cache(
        "example",
        "group",
        "group-a",
        "expired",
        "value",
        datetime.now(UTC) - timedelta(seconds=1),
    )

    assert database.get_cache("example", "group", "group-a", "key") == {"enabled": True}
    assert database.get_cache("example", "group", "group-a", "expired") is None


def test_plugin_settings_are_persisted(tmp_path) -> None:
    database = PluginDatabase(tmp_path / "rennebot.sqlite3")
    database.initialize()

    database.set_setting("admin_user_ids", ["user-a"])

    assert database.get_setting("admin_user_ids") == ["user-a"]
    assert database.get_setting("missing", []) == []


def test_external_database_tool_recovers_administrators(tmp_path) -> None:
    database_path = tmp_path / "rennebot.sqlite3"
    script = Path(__file__).parents[1] / "tools" / "rennebot-db.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--database",
            str(database_path),
            "set",
            "admins",
            "recovered-admin",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Updated admins." in result.stdout
    assert PluginDatabase(database_path).get_setting("admin_user_ids") == ["recovered-admin"]
