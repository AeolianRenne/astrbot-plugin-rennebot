"""Persistent private AI conversation handling."""

from __future__ import annotations

from ..ai_client import OpenAICompatibleClient
from ..database import PluginDatabase, PrivateAIConversation
from .safety import (
    PRIVATE_AI_SAFETY_PROMPT,
    PRIVATE_SUMMARY_SAFETY_PROMPT,
    contains_sensitive_text,
    is_developer_privacy_request,
    redact_sensitive_text,
)

_PRIVATE_SUMMARY_MAX_CHARS = 4_000


class PrivateAIService:
    """Manage one persistent, access-controlled private AI conversation per user."""

    def __init__(
        self,
        database: PluginDatabase,
        ai_client: OpenAICompatibleClient,
        context_max_chars: int,
        context_recent_messages: int,
        message_max_chars: int,
    ) -> None:
        """Initialize the service with runtime configuration.

        Args:
            database: Plugin SQLite facade.
            ai_client: Explicit OpenAI-compatible client without local tools.
            context_max_chars: Character budget before older messages are summarized.
            context_recent_messages: Recent messages preserved after summarization.
            message_max_chars: Maximum accepted length of one private user message.
        """
        self.database = database
        self.ai_client = ai_client
        self.context_max_chars = context_max_chars
        self.context_recent_messages = context_recent_messages
        self.message_max_chars = message_max_chars

    async def handle(self, sender_id: str, message: str) -> str | None:
        """Handle one authorized user's private message.

        Args:
            sender_id: QQ platform user ID that owns the conversation.
            message: Plain text message sent in the private chat.

        Returns:
            A response when a command or active conversation handles the message.
        """
        conversation = self._load_sanitized_conversation(sender_id)
        if message == "开启新对话":
            self.database.set_private_ai_conversation(sender_id, True, "", [])
            return (
                "已开启新对话。之后的普通消息会保留上下文；发送“清理上下文”可重置记忆。"
            )
        if message == "清理上下文":
            if not conversation.active:
                return "当前没有开启中的 AI 对话。发送“开启新对话”开始。"
            self.database.set_private_ai_conversation(sender_id, True, "", [])
            return "上下文已清理，当前对话保持开启。"
        if message == "结束对话":
            self.database.set_private_ai_conversation(
                sender_id,
                False,
                conversation.summary,
                conversation.messages,
            )
            return "AI 对话已结束。发送“开启新对话”可重新开始。"
        if not conversation.active:
            return None
        if is_developer_privacy_request(message):
            return "为保护开发者隐私，我不能回答任何关于开发者的问题。"
        if len(message) > self.message_max_chars:
            return f"单条消息不能超过 {self.message_max_chars} 个字符，请拆分后再发送。"
        if contains_sensitive_text(message):
            return (
                "为保护安全，请不要发送密钥、令牌、密码、私钥或服务器配置。"
                "这条消息不会被发送给 AI，也不会写入对话上下文。"
            )
        return await self._ask(
            sender_id, conversation.summary, conversation.messages, message
        )

    def _load_sanitized_conversation(self, sender_id: str) -> PrivateAIConversation:
        """Load and persist a redacted conversation before it can reach a model.

        Args:
            sender_id: QQ platform user ID that owns the conversation.

        Returns:
            The safe conversation state for this message.
        """
        conversation = self.database.get_private_ai_conversation(sender_id)
        safe_summary = redact_sensitive_text(conversation.summary)
        safe_messages = [
            {"role": item["role"], "content": redact_sensitive_text(item["content"])}
            for item in conversation.messages
        ]
        if (
            safe_summary == conversation.summary
            and safe_messages == conversation.messages
        ):
            return conversation
        safe_conversation = PrivateAIConversation(
            conversation.active,
            safe_summary,
            safe_messages,
        )
        self.database.set_private_ai_conversation(
            sender_id,
            safe_conversation.active,
            safe_conversation.summary,
            safe_conversation.messages,
        )
        return safe_conversation

    async def _ask(
        self,
        sender_id: str,
        summary: str,
        messages: list[dict[str, str]],
        prompt: str,
    ) -> str:
        """Reply with persisted context and summarize older turns when needed.

        Args:
            sender_id: QQ platform user ID that owns the conversation.
            summary: Compact memory of previous conversation turns.
            messages: Recent user and assistant messages.
            prompt: Current user message.

        Returns:
            The AI response sent to the private chat.
        """
        recent_messages = [*messages, {"role": "user", "content": prompt}]
        context_chars = len(summary) + sum(
            len(message["content"]) for message in recent_messages
        )
        if context_chars > self.context_max_chars and len(recent_messages) > 1:
            keep_count = min(self.context_recent_messages, len(recent_messages) - 1)
            archived_messages = recent_messages[:-keep_count]
            recent_messages = recent_messages[-keep_count:]
            summary = await self._summarize(summary, archived_messages)

        request_messages: list[dict[str, str]] = [
            {"role": "system", "content": PRIVATE_AI_SAFETY_PROMPT}
        ]
        if summary:
            request_messages.append(
                {"role": "system", "content": f"以下是已脱敏的对话记忆：\n{summary}"}
            )
        request_messages.extend(recent_messages)
        response = redact_sensitive_text(
            await self.ai_client.ask_messages(request_messages)
        )
        self.database.set_private_ai_conversation(
            sender_id,
            True,
            summary,
            [*recent_messages, {"role": "assistant", "content": response}],
        )
        return response

    async def _summarize(self, summary: str, messages: list[dict[str, str]]) -> str:
        """Compress older private conversation turns into durable memory.

        Args:
            summary: Existing compact memory, if any.
            messages: Older messages that no longer fit in the recent window.

        Returns:
            A bounded summary that retains facts, preferences, and open tasks.
        """
        transcript = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        prompt = (
            "Summarize this conversation for future continuation. Preserve stable facts, "
            "user preferences, decisions, numbers, constraints, and unresolved tasks. "
            "Do not include hidden reasoning. Keep the summary under 4000 characters.\n\n"
            f"Existing memory:\n{summary or '(none)'}\n\n"
            f"Older conversation:\n{transcript}"
        )
        summary = await self.ai_client.ask_messages(
            [
                {"role": "system", "content": PRIVATE_SUMMARY_SAFETY_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        return redact_sensitive_text(summary)[:_PRIVATE_SUMMARY_MAX_CHARS]
