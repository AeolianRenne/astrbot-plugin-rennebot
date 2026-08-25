"""Shared compatibility helpers for supported QQ message platforms."""

from __future__ import annotations


def platform_label(platform_name: str) -> str:
    """Return a Chinese label for a supported QQ platform.

    Args:
        platform_name: AstrBot platform adapter type name.

    Returns:
        A user-facing platform label.
    """
    if platform_name == "aiocqhttp":
        return "QQ 个人号（OneBot v11）"
    return "QQ 官方机器人"


def is_onebot_self_message(
    platform_name: str, sender_id: str, self_id: str | None
) -> bool:
    """Check whether a OneBot event was sent by the connected QQ account.

    Args:
        platform_name: AstrBot platform adapter type name.
        sender_id: Event sender platform ID.
        self_id: Connected account ID supplied by the OneBot implementation.

    Returns:
        Whether the event must be ignored to prevent a self-reply loop.
    """
    return platform_name == "aiocqhttp" and bool(self_id) and sender_id == self_id
