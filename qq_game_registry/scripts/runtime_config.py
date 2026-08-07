"""Private administrator configuration commands."""

from __future__ import annotations

from collections.abc import Callable

from ..commands import CommandError, parse_runtime_config_command
from ..database import PluginDatabase


def handle_config_message(
    sender_id: str,
    message: str,
    database: PluginDatabase,
    setting_ids: Callable[[str], set[str]],
) -> str:
    """Update or display private runtime configuration for an administrator.

    Args:
        sender_id: QQ sender platform ID.
        message: Private command text.
        database: Plugin SQLite facade.
        setting_ids: Callback that returns IDs stored for one configuration key.

    Returns:
        User-facing configuration result.
    """
    if sender_id not in setting_ids("admin_user_ids"):
        return "你还不是 RenneBot 管理员。"
    try:
        command = parse_runtime_config_command(message)
    except CommandError as error:
        return str(error)
    if command is None:
        return "未知配置指令。"
    if command.action == "show":
        private_users = ", ".join(sorted(setting_ids("ai_private_user_ids")))
        groups = ", ".join(sorted(setting_ids("ai_group_ids")))
        admins = ", ".join(sorted(setting_ids("admin_user_ids")))
        return (
            f"AI 私聊白名单：{private_users or '（未配置）'}\n"
            f"AI 群聊白名单：{groups or '（未配置）'}\n"
            f"机器人管理员：{admins or '（未配置）'}"
        )
    if command.setting_key == "admin_user_ids" and sender_id not in command.values:
        return "管理员列表必须保留你自己的 ID，避免失去管理权限。"
    database.set_setting(command.setting_key or "", sorted(set(command.values)))
    setting_names = {
        "ai_private_user_ids": "AI 私聊白名单",
        "ai_group_ids": "AI 群聊白名单",
        "admin_user_ids": "机器人管理员",
    }
    setting_name = setting_names.get(command.setting_key or "", "配置")
    return f"已更新{setting_name}，共 {len(command.values)} 个 ID。"
