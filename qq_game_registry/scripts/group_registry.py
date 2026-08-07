"""Group game registry commands and one-shot group AI handling."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from astrbot.api.event import AstrMessageEvent

from ..commands import CommandError, CommandKind, ParsedCommand, parse_group_command
from ..database import PluginDatabase


async def handle_group_message(
    event: AstrMessageEvent,
    group_id: str,
    sender_id: str,
    message: str,
    database: PluginDatabase,
    setting_ids: Callable[[str], set[str]],
    ask_ai: Callable[[str], Awaitable[str]],
) -> str | None:
    """Handle an addressed command in one QQ group.

    Args:
        event: Incoming QQ Official event.
        group_id: QQ group platform ID.
        sender_id: QQ sender platform ID.
        message: Plain message text.
        database: Plugin SQLite facade.
        setting_ids: Callback that returns IDs stored for one configuration key.
        ask_ai: Callback that makes one authorized group AI request.

    Returns:
        Reply text, or None when a normal group message should be ignored.
    """
    try:
        command = parse_group_command(message)
    except CommandError as error:
        return str(error)
    if command is None:
        return None
    if command.kind == CommandKind.AI:
        if group_id not in setting_ids("ai_group_ids"):
            return "此群未启用 AI。"
        return await ask_ai(command.prompt or "")
    if command.kind == CommandKind.HELP:
        return help_text()
    if command.kind == CommandKind.IDENTITY:
        return f"群 ID 是：{group_id}\n你的 UserID 是：{sender_id}"
    return handle_registry_command(
        event, group_id, sender_id, command, database, setting_ids
    )


def handle_registry_command(
    event: AstrMessageEvent,
    group_id: str,
    sender_id: str,
    command: ParsedCommand,
    database: PluginDatabase,
    setting_ids: Callable[[str], set[str]],
) -> str:
    """Perform a validated group game registry command.

    Args:
        event: Incoming QQ Official event.
        group_id: QQ group platform ID.
        sender_id: QQ sender platform ID.
        command: Parsed registry command.
        database: Plugin SQLite facade.
        setting_ids: Callback that returns IDs stored for one configuration key.

    Returns:
        User-facing operation result.
    """
    if command.kind == CommandKind.RECORD_GAME_ID:
        display_name = event.get_sender_name() or sender_id
        database.upsert_game_id(
            group_id,
            sender_id,
            display_name,
            command.game_name or "",
            command.game_id or "",
        )
        return f"已记录 {display_name} 的《{command.game_name}》ID：{command.game_id}。"
    if command.kind == CommandKind.QUERY_GAME_ID:
        records = database.list_game_ids(group_id, command.game_name or "")
        if not records:
            return f"本群还没有《{command.game_name}》的登记记录。"
        lines = [f"《{command.game_name}》群友 ID："]
        lines.extend(f"{record.display_name}：{record.game_id}" for record in records)
        return "\n".join(lines)
    if command.kind == CommandKind.DELETE_GAME_ID:
        target_user_id = command.target_user_id or sender_id
        if target_user_id != sender_id and sender_id not in setting_ids(
            "admin_user_ids"
        ):
            return "只能删除自己的记录；机器人管理员可以指定 QQ 号删除。"
        deleted = database.delete_game_id(
            group_id,
            target_user_id,
            command.game_name or "",
        )
        return "已删除记录。" if deleted else "没有找到可删除的记录。"
    return "未知指令。"


def help_text() -> str:
    """Return the group command reference.

    Returns:
        Human-readable command list.
    """
    return (
        "可用指令：\n"
        "/记录游戏id <游戏名> <数字ID>\n"
        "/查询群友id <游戏名>\n"
        "/删除游戏id <游戏名> [QQ号]\n"
        "/ai <问题>（仅已授权群）\n"
        "/renne-id"
    )
