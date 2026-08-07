import httpx
import pytest

from qq_game_registry.ai_client import AIRequestError, OpenAICompatibleClient


@pytest.mark.asyncio
async def test_openai_compatible_request(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_BASE", "https://ai.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_MODEL", "example-model")

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://ai.example/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "你好"}}]},
        )

    client = OpenAICompatibleClient(transport=httpx.MockTransport(handler))

    assert await client.ask("测试") == "你好"


@pytest.mark.asyncio
async def test_invalid_ai_response_is_safe(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_BASE", "https://ai.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_MODEL", "example-model")
    client = OpenAICompatibleClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={}))
    )

    with pytest.raises(AIRequestError):
        await client.ask("测试")
