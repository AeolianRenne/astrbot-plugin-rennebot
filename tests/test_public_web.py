"""Tests for credential-free public-page extraction safeguards."""

import pytest

from qq_game_registry.scripts.public_web import (
    PublicWebExtractor,
    PublicWebResponse,
    is_public_http_url,
)


class FakeFetcher:
    """Return deterministic pages and retain requested URLs."""

    def __init__(self, responses: dict[str, PublicWebResponse]) -> None:
        """Initialize the fake response map.

        Args:
            responses: Response returned for each requested public URL.
        """
        self.responses = responses
        self.urls: list[str] = []

    async def __call__(self, url: str) -> PublicWebResponse:
        """Record and return the mapped response.

        Args:
            url: Requested page URL.

        Returns:
            Configured response.
        """
        self.urls.append(url)
        return self.responses[url]


@pytest.mark.asyncio
async def test_extracts_visible_text_after_a_safe_redirect() -> None:
    fetcher = FakeFetcher(
        {
            "https://example.com/start": PublicWebResponse(
                302, {"location": "/article"}, b""
            ),
            "https://example.com/article": PublicWebResponse(
                200,
                {"content-type": "text/html; charset=utf-8"},
                b"<html><head><title>Hidden</title></head><body>Public <b>article</b><script>secret()</script></body></html>",
            ),
        }
    )

    content = await PublicWebExtractor(fetcher=fetcher).extract(
        "https://example.com/start"
    )

    assert content == "Public\narticle"
    assert fetcher.urls == ["https://example.com/start", "https://example.com/article"]


@pytest.mark.asyncio
async def test_rejects_private_redirects_before_fetching_them() -> None:
    fetcher = FakeFetcher(
        {
            "https://example.com/start": PublicWebResponse(
                302, {"location": "http://127.0.0.1/admin"}, b""
            )
        }
    )

    content = await PublicWebExtractor(fetcher=fetcher).extract(
        "https://example.com/start"
    )

    assert content is None
    assert fetcher.urls == ["https://example.com/start"]


@pytest.mark.asyncio
async def test_rejects_non_html_and_oversized_responses() -> None:
    fetcher = FakeFetcher(
        {
            "https://example.com/document.pdf": PublicWebResponse(
                200, {"content-type": "application/pdf"}, b"%PDF"
            ),
            "https://example.com/large": PublicWebResponse(
                200, {"content-type": "text/html"}, b"x" * 101
            ),
        }
    )
    extractor = PublicWebExtractor(max_bytes=100, fetcher=fetcher)

    assert await extractor.extract("https://example.com/document.pdf") is None
    assert await extractor.extract("https://example.com/large") is None


@pytest.mark.asyncio
async def test_rejects_a_page_that_presents_a_password_login_form() -> None:
    fetcher = FakeFetcher(
        {
            "https://example.com/login": PublicWebResponse(
                200,
                {"content-type": "text/html"},
                b"<form><input type='password'></form>",
            )
        }
    )

    assert await PublicWebExtractor(fetcher=fetcher).extract(
        "https://example.com/login"
    ) is None


def test_rejects_local_credentialed_and_unusual_port_urls() -> None:
    assert is_public_http_url("https://example.com/article")
    assert not is_public_http_url("https://user:pass@example.com/article")
    assert not is_public_http_url("http://127.0.0.1/article")
    assert not is_public_http_url("http://169.254.169.254/latest/meta-data")
    assert not is_public_http_url("https://example.com:8080/article")
