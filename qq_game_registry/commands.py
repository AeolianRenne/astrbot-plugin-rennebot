"""Command parsing that can be tested independently from AstrBot."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class CommandKind(StrEnum):
    """Supported group command kinds."""

    RECORD_GAME_ID = "record_game_id"
    QUERY_GAME_ID = "query_game_id"
    DELETE_GAME_ID = "delete_game_id"
    AI = "ai"
    HELP = "help"
    IDENTITY = "identity"


def message_text_from_plain_components(
    plain_texts: Iterable[str], fallback: str
) -> str:
    """Prefer text parsed from message components over a raw event string.

    Args:
        plain_texts: Text values from plain message components.
        fallback: The event's raw text when no plain component is available.

    Returns:
        Normalized text suitable for command matching.
    """
    text = " ".join(item for item in plain_texts if item).strip()
    return text or fallback.strip()


@dataclass(frozen=True)
class ParsedCommand:
    """A validated command with its parsed arguments."""

    kind: CommandKind
    game_name: str | None = None
    game_id: str | None = None
    target_user_id: str | None = None
    prompt: str | None = None


class CommandError(ValueError):
    """Raised when an addressed command has invalid syntax."""


_NUMBER = re.compile(r"^\d{1,64}$")
_CONFIG_KEYS = {
    "ai-users": "ai_private_user_ids",
    "ai-groups": "ai_group_ids",
    "admins": "admin_user_ids",
}


@dataclass(frozen=True)
class RuntimeConfigCommand:
    """A private administrator configuration command."""

    action: str
    setting_key: str | None = None
    values: tuple[str, ...] = ()


def parse_group_command(message: str) -> ParsedCommand | None:
    """Parse an AstrBot-normalized group message.

    Args:
        message: The plain textual message after AstrBot has separated QQ mentions.

    Returns:
        A parsed command, or ``None`` when the message is not a bot command.

    Raises:
        CommandError: If a slash command has invalid arguments.
    """
    text = message.strip()
    if not text.startswith("/"):
        return None

    command, _, remainder = text[1:].partition(" ")
    command = command.casefold()
    remainder = remainder.strip()

    if command in {"帮助", "help"}:
        return ParsedCommand(CommandKind.HELP)
    if command == "renne-id":
        if remainder:
            raise CommandError("用法：/renne-id")
        return ParsedCommand(CommandKind.IDENTITY)
    if command == "ai":
        if not remainder:
            raise CommandError("用法：/ai <问题>")
        return ParsedCommand(CommandKind.AI, prompt=remainder)
    if command == "记录游戏id":
        parts = remainder.split()
        if len(parts) != 2:
            raise CommandError("用法：/记录游戏id <游戏名> <数字ID>")
        if not _NUMBER.fullmatch(parts[1]):
            raise CommandError("数字ID必须由 1 到 64 位数字组成。")
        return ParsedCommand(
            CommandKind.RECORD_GAME_ID,
            game_name=parts[0],
            game_id=parts[1],
        )
    if command in {"查询群友id", "查询游戏id"}:
        if not remainder or " " in remainder:
            raise CommandError("用法：/查询群友id <游戏名>")
        return ParsedCommand(CommandKind.QUERY_GAME_ID, game_name=remainder)
    if command in {"删除游戏id", "删除群友id"}:
        parts = remainder.split()
        if len(parts) not in {1, 2}:
            raise CommandError("用法：/删除游戏id <游戏名> [QQ号]")
        if len(parts) == 2 and not _NUMBER.fullmatch(parts[1]):
            raise CommandError("QQ号只能包含数字。")
        return ParsedCommand(
            CommandKind.DELETE_GAME_ID,
            game_name=parts[0],
            target_user_id=parts[1] if len(parts) == 2 else None,
        )
    return None


def parse_runtime_config_command(message: str) -> RuntimeConfigCommand | None:
    """Parse a private administrator configuration command.

    Args:
        message: Private plain text message.

    Returns:
        Parsed configuration command, or None when not a configuration command.

    Raises:
        CommandError: If the configuration command is malformed.
    """
    parts = message.strip().split(maxsplit=3)
    if not parts or parts[0] != "/renne-config":
        return None
    if len(parts) == 2 and parts[1] == "show":
        return RuntimeConfigCommand("show")
    if len(parts) != 4 or parts[2] != "set" or parts[1] not in _CONFIG_KEYS:
        raise CommandError(
            "用法：/renne-config show 或 /renne-config "
            "<ai-users|ai-groups|admins> set <id,id>"
        )
    values = tuple(item.strip() for item in parts[3].split(",") if item.strip())
    if not values or any(
        any(character.isspace() for character in item) for item in values
    ):
        raise CommandError("配置 ID 必须是非空、以英文逗号分隔的列表。")
    return RuntimeConfigCommand("set", _CONFIG_KEYS[parts[1]], values)
