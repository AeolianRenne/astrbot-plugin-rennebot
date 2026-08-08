import asyncio
import json

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


class FakePageExtractor:
    """Keep existing research-task tests independent from page retrieval."""

    def __init__(self) -> None:
        """Initialize an empty extraction log."""
        self.urls: list[str] = []

    async def extract(self, url: str) -> str | None:
        """Return no page text so the search summary remains the evidence.

        Args:
            url: Source URL to record.

        Returns:
            No extracted content.
        """
        self.urls.append(url)
        return None


class FakeAIClient:
    """Return a deterministic answer without an external model request."""

    async def ask_messages(self, _: list[dict[str, str]]) -> str:
        """Return a fixed model response.

        Returns:
            Fixed text used by research-task tests.
        """
        return "已根据来源完成整理。"


class PlanningAIClient:
    """Return a follow-up search plan, then a deterministic final answer."""

    def __init__(self) -> None:
        """Initialize the request counter."""
        self.calls = 0

    async def ask_messages(self, _: list[dict[str, str]]) -> str:
        """Return bounded planner JSON before the final answer.

        Returns:
            Planner JSON for the first call and an answer afterwards.
        """
        self.calls += 1
        if self.calls == 1:
            return '{"queries": ["Kimi API pricing", "MiniMax API pricing"]}'
        return "已根据来源完成整理。"


class FakeSearchProvider:
    """Record queries and return one safe public source."""

    def __init__(self) -> None:
        """Initialize an empty query log."""
        self.queries: list[str] = []
        self.domain_filters: list[tuple[str, ...]] = []

    async def search(
        self,
        query: str,
        _: int,
        allowed_domains: tuple[str, ...] = (),
    ) -> list[SearchResult]:
        """Return one deterministic source.

        Args:
            query: Research query to record.
            _: Ignored maximum source count.
            allowed_domains: Official domain constraint to record.

        Returns:
            One public source.
        """
        self.queries.append(query)
        self.domain_filters.append(allowed_domains)
        return [
            SearchResult(
                "公开来源",
                "https://example.com/research",
                "可引用的检索摘要。",
                "2026-08-08",
            )
        ]


class SlowSearchProvider:
    """Wait long enough to exercise the whole-task deadline."""

    async def search(
        self,
        _: str,
        __: int,
        ___: tuple[str, ...] = (),
    ) -> list[SearchResult]:
        """Delay beyond the configured task budget.

        Args:
            _: Ignored query.
            __: Ignored maximum result count.
            ___: Ignored official-domain constraint.

        Returns:
            This method does not normally return before cancellation.
        """
        await asyncio.sleep(1)
        return []


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
    research = ResearchTaskService(
        database,
        FakeAIClient(),
        search,
        FakePageExtractor(),
        10_000,
        8,
        6,
        6,
        2,
        12,
        900,
        90,
    )
    return PrivateAIService(
        database, FakeAIClient(), 10_000, 8, 8_000, research
    ), search


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

    assert "请先发送“结束当前任务”或“结束任务”" in await service.handle(
        "user-a", "开启新对话"
    )
    assert "当前任务已结束" in await service.handle("user-a", "结束任务")
    assert "已开启新对话" in await service.handle("user-a", "开启新对话")


@pytest.mark.asyncio
async def test_task_start_alias_sends_status_before_external_requests(tmp_path) -> None:
    service, search = make_service(tmp_path)
    status_updates: list[str] = []

    async def notify_started() -> None:
        """Record the point at which a valid task was persisted."""
        status_updates.append("started")

    reply = await service.handle(
        "user-a",
        "开启任务：整理近期 AI 新闻",
        notify_started,
    )

    assert status_updates == ["started"]
    assert len(search.queries) == 1
    assert "已开始任务" not in reply


@pytest.mark.asyncio
async def test_research_task_caches_results_per_user_and_keeps_goal_on_clear(
    tmp_path,
) -> None:
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
async def test_pricing_task_plans_official_queries_for_named_providers(
    tmp_path,
) -> None:
    service, search = make_service(tmp_path)

    await service.handle(
        "user-a",
        "开始任务：整理 DeepSeek、Qwen、Kimi、MiniMax 和 GLM 的 API 定价",
    )

    assert len(search.queries) == 6
    assert search.domain_filters[:5] == [
        ("api-docs.deepseek.com",),
        ("help.aliyun.com", "dashscope.aliyuncs.com"),
        ("platform.kimi.com", "kimi.com"),
        ("platform.minimaxi.com", "platform.minimax.io"),
        ("docs.bigmodel.cn", "open.bigmodel.cn"),
    ]
    assert all("官方 API 定价 服务端接入" in query for query in search.queries[:5])


@pytest.mark.asyncio
async def test_task_operation_budget_caps_searches_and_page_extractions(
    tmp_path,
) -> None:
    """Keep a task turn within its configured logical public-web budget."""
    database = PluginDatabase(tmp_path / "rennebot.sqlite3")
    database.initialize()
    search = FakeSearchProvider()
    pages = FakePageExtractor()
    research = ResearchTaskService(
        database,
        FakeAIClient(),
        search,
        pages,
        10_000,
        8,
        6,
        6,
        2,
        2,
        900,
        90,
    )

    await research.start(
        "user-a",
        "整理 DeepSeek、Qwen、Kimi、MiniMax 和 GLM 的 API 定价",
    )

    assert len(search.queries) == 2
    assert pages.urls == []


@pytest.mark.asyncio
async def test_generic_task_uses_evidence_to_plan_a_second_search_round(
    tmp_path,
) -> None:
    """Plan targeted follow-up searches without giving the model direct web access."""
    database = PluginDatabase(tmp_path / "rennebot.sqlite3")
    database.initialize()
    search = FakeSearchProvider()
    research = ResearchTaskService(
        database,
        PlanningAIClient(),
        search,
        FakePageExtractor(),
        10_000,
        8,
        6,
        6,
        2,
        12,
        900,
        90,
    )

    await research.start("user-a", "整理公开的 AI 模型服务和 API 定价")

    assert len(search.queries) == 3
    assert search.queries[1:] == ["Kimi API pricing", "MiniMax API pricing"]


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


@pytest.mark.asyncio
async def test_tavily_applies_official_domain_filter(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    request_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_body
        request_body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Unexpected domain",
                        "url": "https://example.com/pricing",
                        "content": "Not an official DeepSeek source.",
                    }
                ]
            },
        )

    provider = TavilySearchProvider(transport=httpx.MockTransport(handler))

    assert (
        await provider.search("DeepSeek 官方 API 定价", 1, ("api-docs.deepseek.com",))
        == []
    )
    assert request_body["include_domains"] == ["api-docs.deepseek.com"]


@pytest.mark.asyncio
async def test_task_deadline_returns_a_safe_status_without_a_model_answer(
    tmp_path,
) -> None:
    database = PluginDatabase(tmp_path / "rennebot.sqlite3")
    database.initialize()
    research = ResearchTaskService(
        database,
        FakeAIClient(),
        SlowSearchProvider(),
        FakePageExtractor(),
        10_000,
        8,
        6,
        6,
        2,
        12,
        900,
        0.01,
    )

    reply = await research.start("user-a", "整理公开模型定价")

    assert "未完成" in reply
    assert "未使用不完整资料生成结论" in reply
