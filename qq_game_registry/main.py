"""AstrBot entrypoint for controlled QQ Official Bot message handling."""

from __future__ import annotations

import os

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.core.message.components import At, Plain
from astrbot.core.star.star_tools import StarTools

from .ai_client import AIConfigurationError, AIRequestError, OpenAICompatibleClient
from .commands import message_text_from_plain_components
from .database import PluginDatabase
from .scripts.group_registry import handle_group_message
from .scripts.private_ai import PrivateAIService
from .scripts.runtime_config import handle_config_message

_PRIVATE_CONTEXT_MAX_CHARS_DEFAULT = 120_000
_PRIVATE_CONTEXT_RECENT_MESSAGES_DEFAULT = 24
_PRIVATE_MESSAGE_MAX_CHARS_DEFAULT = 8_000


def _configured_ids(variable: str) -> set[str]:
    """Parse a comma-separated platform-ID environment variable.

    Args:
        variable: Environment variable name.

    Returns:
        Non-empty platform IDs from the setting.
    """
    return {item.strip() for item in os.getenv(variable, "").split(",") if item.strip()}


def _positive_int(variable: str, default: int) -> int:
    """Read a positive integer environment setting with a safe fallback.

    Args:
        variable: Environment variable name.
        default: Value used for missing or invalid input.

    Returns:
        A positive integer.
    """
    try:
        value = int(os.getenv(variable, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


class Main(Star):
    """Route QQ Official events to RenneBot feature modules."""

    def __init__(self, context: Context) -> None:
        """Initialize infrastructure and feature services.

        Args:
            context: AstrBot plugin context.
        """
        super().__init__(context)
        self.database = PluginDatabase(
            StarTools.get_data_dir("qq_game_registry") / "rennebot.sqlite3"
        )
        self.database.initialize()
        if self.database.get_setting("admin_user_ids") is None:
            bootstrap_admins = _configured_ids("RENNEBOT_BOOTSTRAP_ADMIN_IDS")
            if bootstrap_admins:
                self.database.set_setting("admin_user_ids", sorted(bootstrap_admins))
        self.ai_client = OpenAICompatibleClient()
        self.private_ai = PrivateAIService(
            self.database,
            self.ai_client,
            _positive_int(
                "AI_PRIVATE_CONTEXT_MAX_CHARS", _PRIVATE_CONTEXT_MAX_CHARS_DEFAULT
            ),
            _positive_int(
                "AI_PRIVATE_CONTEXT_RECENT_MESSAGES",
                _PRIVATE_CONTEXT_RECENT_MESSAGES_DEFAULT,
            ),
            _positive_int(
                "AI_PRIVATE_MESSAGE_MAX_CHARS", _PRIVATE_MESSAGE_MAX_CHARS_DEFAULT
            ),
        )

    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    @filter.platform_adapter_type(filter.PlatformAdapterType.QQOFFICIAL)
    async def route_message(self, event: AstrMessageEvent):
        """Route permitted QQ messages and block AstrBot's default LLM flow.

        Args:
            event: Incoming QQ Official message event.

        Yields:
            A plain response when a command or authorized AI request needs one.
        """
        try:
            group_id = event.get_group_id()
            sender_id = event.get_sender_id()
            messages = event.get_messages()
            message = message_text_from_plain_components(
                (
                    component.text
                    for component in messages
                    if isinstance(component, Plain)
                ),
                event.message_str,
            )
            if group_id:
                response = (
                    await handle_group_message(
                        event,
                        group_id,
                        sender_id,
                        message,
                        self.database,
                        self._setting_ids,
                        self._ask_group_ai,
                    )
                    if any(isinstance(component, At) for component in messages)
                    else None
                )
            elif message == "/renne-id":
                response = f"你的 UserID 是：{sender_id}"
            elif message.startswith("/renne-config"):
                response = handle_config_message(
                    sender_id,
                    message,
                    self.database,
                    self._setting_ids,
                )
            elif sender_id in self._setting_ids("ai_private_user_ids") and message:
                response = await self.private_ai.handle(sender_id, message)
            else:
                response = None
            if response:
                yield event.plain_result(response)
        except Exception as error:
            self.logger.exception("qq_game_registry message handling failed: %s", error)
            yield event.plain_result("处理消息时发生了错误，请稍后再试。")
        finally:
            event.stop_event()

    def _setting_ids(self, key: str) -> set[str]:
        """Read a platform-ID setting stored in SQLite.

        Args:
            key: Plugin setting key.

        Returns:
            String platform IDs stored under the key.
        """
        value = self.database.get_setting(key, [])
        if not isinstance(value, list):
            return set()
        return {item for item in value if isinstance(item, str)}

    async def _ask_group_ai(self, prompt: str) -> str:
        """Return a safe result from the configured one-shot group AI endpoint.

        Args:
            prompt: Authorized group prompt.

        Returns:
            AI response or a safe configuration or request error.
        """
        try:
            return await self.ai_client.ask(prompt)
        except (AIConfigurationError, AIRequestError) as error:
            return str(error)
