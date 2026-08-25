"""Tests for the optional Global BanPick command client."""

from __future__ import annotations

import asyncio
from typing import Any

from qq_game_registry.scripts.banpick import BanpickService


class StubBanpickService(BanpickService):
    """Record BanPick requests without connecting to the standalone service."""

    def __init__(self) -> None:
        self.base_url = "http://banpick.test"
        self.api_key = "test-key"
        self.requests: list[tuple[str, str, dict[str, Any] | None]] = []

    async def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.requests.append((method, path, payload))
        return {"series": {"code": "ABC123", "best_of": 3}}


def test_admin_can_extend_series_format() -> None:
    """The administrator-only command calls the internal format endpoint."""
    service = StubBanpickService()

    reply = asyncio.run(service.handle("/BP 赛制 ABC123 BO3", "admin", {"admin"}))

    assert reply == "赛事 ABC123 已调整为 BO3。"
    assert service.requests == [
        ("POST", "/api/internal/series/ABC123/format", {"best_of": 3})
    ]


def test_non_admin_cannot_extend_series_format() -> None:
    """Format changes stay restricted to configured bot administrators."""
    service = StubBanpickService()

    reply = asyncio.run(service.handle("/BP 赛制 ABC123 BO5", "member", {"admin"}))

    assert reply == "只有机器人管理员可以调整赛事赛制。"
    assert service.requests == []
