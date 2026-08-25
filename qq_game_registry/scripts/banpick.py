"""Optional client for an independently deployed Global BanPick service."""

from __future__ import annotations

import os
from typing import Any

import httpx


class BanpickService:
    """Translate QQ group commands into calls to the standalone BanPick API."""

    def __init__(self) -> None:
        """Load optional service connectivity settings."""
        self.base_url = os.getenv("BANPICK_INTERNAL_URL", "").rstrip("/")
        self.api_key = os.getenv("BANPICK_BOT_API_KEY", "")

    async def handle(
        self, message: str, sender_id: str, admin_ids: set[str]
    ) -> str | None:
        """Handle a slash BP command or ignore unrelated messages.

        Args:
            message: Plain QQ group message text.
            sender_id: QQ platform ID of the command sender.
            admin_ids: Configured RenneBot administrator platform IDs.

        Returns:
            A Chinese reply for a BP command, or None for a different command.
        """
        parts = message.strip().split()
        if not parts or parts[0].casefold() != "/bp":
            return None
        if not self.base_url or not self.api_key:
            return "BP 服务尚未配置，请联系机器人管理员。"
        if len(parts) < 2:
            return self._usage()
        command = parts[1]
        try:
            if command == "创建":
                return await self._create(parts[2:])
            if command == "状态" and len(parts) == 3:
                state = await self._request("GET", f"/api/internal/series/{parts[2]}")
                return self._state_text(state)
            if command == "下一局" and len(parts) == 3:
                state = await self._request("POST", f"/api/internal/series/{parts[2]}/next")
                return f"已创建 {state['series']['code']} 的第 {state['game']['number']} 局，请双方重新确认准备。"
            if command == "结束" and len(parts) == 3:
                state = await self._request("POST", f"/api/internal/series/{parts[2]}/end")
                return f"赛事 {state['series']['code']} 已结束。"
            if command == "更新英雄" and len(parts) == 2:
                if sender_id not in admin_ids:
                    return "只有机器人管理员可以更新英雄资料。"
                result = await self._request("POST", "/api/internal/sync")
                return f"英雄资料更新完成：{result['hero_count']} 名英雄，来源 {result['source']}。"
        except httpx.HTTPError:
            return "BP 服务暂时不可用，请稍后再试。"
        except (KeyError, TypeError):
            return "BP 服务返回了无法识别的数据。"
        return self._usage()

    async def _create(self, arguments: list[str]) -> str:
        """Create a default-global series from validated command arguments.

        Args:
            arguments: Arguments after `/BP 创建`.

        Returns:
            Human-readable match links or a usage error.
        """
        best_of = 1
        global_draft = True
        for argument in arguments:
            upper = argument.upper()
            if upper in {"BO1", "BO3", "BO5"}:
                best_of = int(upper[-1])
            elif argument == "常规":
                global_draft = False
            else:
                return "用法：/BP 创建 [BO1|BO3|BO5] [常规]"
        result = await self._request(
            "POST",
            "/api/internal/series",
            {"best_of": best_of, "global_draft": global_draft},
        )
        mode = "全局 BP" if global_draft else "常规 BP"
        return (
            f"已创建赛事 {result['code']}（BO{best_of}，{mode}）。\n"
            f"蓝色方：{result['blue']}\n红色方：{result['red']}\n观战：{result['spectator']}"
        )

    async def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call the service and turn API errors into HTTP exceptions.

        Args:
            method: HTTP method.
            path: Service-relative API path.
            payload: Optional JSON request body.

        Returns:
            JSON response object.
        """
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.request(
                method,
                f"{self.base_url}{path}",
                headers={"X-Banpick-Api-Key": self.api_key},
                json=payload,
            )
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise TypeError("expected object response")
        return value

    @staticmethod
    def _state_text(state: dict[str, Any]) -> str:
        """Format compact Chinese series status text.

        Args:
            state: Service state response.

        Returns:
            Chinese status message.
        """
        series = state["series"]
        game = state["game"]
        current = state.get("current")
        if current:
            team = "蓝色方" if current["team"] == "blue" else "红色方"
            action = "禁用" if current["kind"] == "ban" else "选择"
            detail = f"当前为{team}{action}。"
        else:
            detail = "当前等待双方准备。" if game["status"] == "waiting_ready" else f"当前状态：{game['status']}。"
        return f"赛事 {series['code']}：第 {game['number']} 局，{detail}"

    @staticmethod
    def _usage() -> str:
        """Return the BP command reference."""
        return (
            "BP 指令：\n/BP 创建 [BO1|BO3|BO5] [常规]\n/BP 状态 <赛事编号>\n"
            "/BP 下一局 <赛事编号>\n/BP 结束 <赛事编号>\n/BP 更新英雄（管理员）"
        )
