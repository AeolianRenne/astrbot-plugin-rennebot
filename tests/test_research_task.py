import httpx
import pytest

from qq_game_registry.database import PluginDatabase
from qq_game_registry.scripts.private_ai import PrivateAIService
from qq_game_registry.scripts.research_task import (
    ResearchConfigurationError,
    ResearchTaskService,
    SearchResult,
    TavilySearchProvider,
)


class FakeAIClient:
    """Return a deterministic answer without an external model request."""

    async def ask_messages(self, _: list[dict[str, str]]) -> str:
        """Return a fixed model response.

        Returns:
            Fixed text used by research-task tests.
        """
        return "已根据来源完成整理。"


class FakeSearchProvider:
    """Record queries and return one safe public source."""

    def __init__(self) -> None:
        """Initialize an empty query log."""
        self.queries: list[str] = []

    async def search(self, query: str, _: int) -> list[SearchResult]:
        """Return one deterministic source.

        Args:
            query: Research query to record.
            _: Ignored maximum source count.

        Returns:
            One public source.
        """
        self.queries.append(query)
        return [
            SearchResult(
                "公开来源",
                "https://example.com/research",
                "可引用的检索摘要。",
                "2026-08-08",
            )
        ]


def make_service(tmp_path):
    """Build private and research services with deterministic dependencies.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Private service and fake search provider for assertions.
    """
    database = PluginDatabase(tmp_path / "rennebot.sqlite3")
    database.initialize()
    search = FakeSearchProvider()
    research = ResearchTaskService(database, FakeAIClient(), search, 10_000, 8, 3, 900)
    return PrivateAIService(database, FakeAIClient(), 10_000, 8, 8_000, research), search


@pytest.mark.asyncio
async def test_conversation_and_research_task_are_mutually_exclusive(tmp_path) -> None:
    service, search = make_service(tmp_path)

    assert "已开启新对话" in await service.handle("user-a", "开启新对话")
    assert "请先发送“结束对话”" in await service.handle(
        "user-a", "开始任务：整理近期 AI 新闻"
    )
    assert search.queries == []

    assert "AI 对话已结束" in await service.handle("user-a", "结束对话")
    research_reply = await service.handle("user-a", "开始任务：整理近期 AI 新闻")
    assert "已开始任务" in research_reply
    assert "https://example.com/research" in research_reply
    assert len(search.queries) == 1

    assert "请先发送“结束当前任务”" in await service.handle("user-a", "开启新对话")
    assert "当前任务已结束" in await service.handle("user-a", "结束当前任务")
    assert "已开启新对话" in await service.handle("user-a", "开启新对话")


@pytest.mark.asyncio
async def test_research_task_caches_results_per_user_and_keeps_goal_on_clear(tmp_path) -> None:
    service, search = make_service(tmp_path)

    await service.handle("user-a", "开始任务：跟踪模型更新")
    assert len(search.queries) == 1
    await service.handle("user-a", "跟踪模型更新")
    assert len(search.queries) == 1
    assert "任务过程上下文已清理" in await service.handle("user-a", "清理上下文")
    await service.handle("user-a", "跟进最新消息")
    assert len(search.queries) == 2

    await service.handle("user-a", "结束当前任务")
    await service.handle("user-a", "开始任务：跟踪模型更新")
    assert len(search.queries) == 3

    await service.handle("user-b", "开始任务：跟踪模型更新")
    assert len(search.queries) == 4


@pytest.mark.asyncio
async def test_unconfigured_tavily_makes_no_http_request(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    called = False

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"results": []})

    provider = TavilySearchProvider(transport=httpx.MockTransport(handler))

    with pytest.raises(ResearchConfigurationError):
        await provider.search("test", 3)

    assert not called
