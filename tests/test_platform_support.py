"""Tests for QQ Official Bot and OneBot compatibility helpers."""

from __future__ import annotations

from qq_game_registry.scripts.platform_support import (
    is_onebot_self_message,
    platform_label,
)


def test_onebot_self_messages_are_ignored_without_affecting_official_events() -> None:
    """Prevent self-reply loops only for the connected personal QQ account."""
    assert is_onebot_self_message("aiocqhttp", "123", "123")
    assert not is_onebot_self_message("aiocqhttp", "456", "123")
    assert not is_onebot_self_message("qq_official", "123", "123")
    assert platform_label("aiocqhttp") == "QQ 个人号（OneBot v11）"
    assert platform_label("qq_official") == "QQ 官方机器人"
