"""OpenAI-compatible chat requests for explicitly authorized messages."""

from __future__ import annotations

import os

import httpx


class AIConfigurationError(RuntimeError):
    """Raised when AI credentials or model settings are not configured."""


class AIRequestError(RuntimeError):
    """Raised when an AI endpoint cannot provide a valid response."""


class OpenAICompatibleClient:
    """Call an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        """Load API configuration from the runtime environment.

        Args:
            transport: Optional HTTPX transport, used by tests.
        """
        self.api_base = os.getenv("OPENAI_API_BASE", "").rstrip("/")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", "")
        self.timeout_seconds = float(os.getenv("AI_TIMEOUT_SECONDS", "30"))
        self.transport = transport

    async def ask(self, prompt: str) -> str:
        """Request one response for one user prompt.

        Args:
            prompt: User text to send to the configured model.

        Returns:
            Non-empty response text.

        Raises:
            AIConfigurationError: If required environment variables are missing.
            AIRequestError: If the endpoint fails or returns an invalid payload.
        """
        return await self.ask_messages([{"role": "user", "content": prompt}])

    async def ask_messages(self, messages: list[dict[str, str]]) -> str:
        """Request one response for an ordered conversation.

        Args:
            messages: OpenAI-compatible system, user, and assistant messages.

        Returns:
            Non-empty response text.

        Raises:
            AIConfigurationError: If required environment variables are missing.
            AIRequestError: If the endpoint fails or returns an invalid payload.
        """
        if not self.api_base or not self.api_key or not self.model:
            raise AIConfigurationError("AI 尚未配置，请联系机器人管理员。")
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": messages,
                    },
                )
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise AIRequestError("AI 服务暂时不可用，请稍后再试。") from error

        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise AIRequestError("AI 服务返回了无法识别的结果。") from error
        if not isinstance(content, str) or not content.strip():
            raise AIRequestError("AI 服务没有返回文本结果。")
        return content.strip()
