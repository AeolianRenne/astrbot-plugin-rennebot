"""Tests for editable runtime allowlist configuration."""

from __future__ import annotations

from qq_game_registry.database import PluginDatabase
from qq_game_registry.scripts.runtime_config import handle_config_message


def test_administrator_can_manage_the_group_response_allowlist(tmp_path) -> None:
    """Keep group-response authorization separate from the narrower AI allowlist."""
    database = PluginDatabase(tmp_path / "rennebot.sqlite3")
    database.initialize()
    database.set_setting("admin_user_ids", ["admin"])

    def setting_ids(key: str) -> set[str]:
        """Read one list-valued setting for the command handler.

        Args:
            key: Plugin setting key.

        Returns:
            Stored string IDs.
        """
        value = database.get_setting(key, [])
        return {item for item in value if isinstance(item, str)}

    reply = handle_config_message(
        "admin",
        "/renne-config groups set group-1,group-2",
        database,
        setting_ids,
    )

    assert reply == "已更新群聊功能白名单，共 2 个 ID。"
    assert setting_ids("enabled_group_ids") == {"group-1", "group-2"}
    assert "群聊功能白名单：group-1, group-2" in handle_config_message(
        "admin", "/renne-config show", database, setting_ids
    )
