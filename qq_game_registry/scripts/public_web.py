"""Safe extraction of text from public web pages used by research tasks."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import urljoin, urlparse

import aiohttp

_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
_SKIPPED_HTML_TAGS = {"head", "noscript", "script", "style", "svg", "template"}


@dataclass(frozen=True)
class PublicWebResponse:
    """A bounded public HTTP response used by the extractor."""

    status: int
    headers: Mapping[str, str]
    body: bytes


class PublicPageExtractor(Protocol):
    """Extract text from a public HTTP(S) page without authentication."""

    async def extract(self, url: str) -> str | None:
        """Return sanitized page text, or ``None`` when it cannot be safely read."""


class _VisibleTextParser(HTMLParser):
    """Collect visible text while ignoring executable and metadata elements."""

    def __init__(self) -> None:
        """Initialize an empty parser."""
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skipped_depth = 0
        self.has_password_input = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track skipped HTML elements.

        Args:
            tag: Element name.
            attrs: Element attributes, which are not used.
        """
        attributes = {name.casefold(): value for name, value in attrs}
        if (
            tag.casefold() == "input"
            and attributes.get("type", "").casefold() == "password"
        ):
            self.has_password_input = True
        if tag.casefold() in _SKIPPED_HTML_TAGS:
            self.skipped_depth += 1

    def handle_endtag(self, tag: str) -> None:
        """Leave a skipped HTML element.

        Args:
            tag: Element name.
        """
        if tag.casefold() in _SKIPPED_HTML_TAGS and self.skipped_depth:
            self.skipped_depth -= 1

    def handle_data(self, data: str) -> None:
        """Collect visible non-empty text.

        Args:
            data: Text node content.
        """
        if not self.skipped_depth:
            normalized = " ".join(unescape(data).split())
            if normalized:
                self.parts.append(normalized)


class _PublicDNSResolver(aiohttp.abc.AbstractResolver):
    """Resolve a hostname once and connect only to its validated public IPs."""

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_UNSPEC,
    ) -> list[aiohttp.abc.ResolveResult]:
        """Resolve and validate every address before a connection is opened.

        Args:
            host: Requested public hostname.
            port: Requested TCP port.
            family: Address family requested by aiohttp.

        Returns:
            Validated address records used directly for the TCP connection.

        Raises:
            OSError: If the hostname does not resolve exclusively to public IPs.
        """
        normalized_host = host.casefold().rstrip(".")
        if normalized_host in {
            "localhost",
            "localhost.localdomain",
        } or normalized_host.endswith(".local"):
            raise OSError("non-public host")
        try:
            if not _is_public_ip(normalized_host):
                raise OSError("non-public host")
        except ValueError:
            pass
        records = await asyncio.get_running_loop().getaddrinfo(
            host,
            port,
            family=family,
            type=socket.SOCK_STREAM,
        )
        resolved: list[aiohttp.abc.ResolveResult] = []
        for address_family, _, _, _, sockaddr in records:
            address = sockaddr[0]
            if not _is_public_ip(address):
                raise OSError("host resolved to a non-public IP")
            resolved.append(
                {
                    "hostname": host,
                    "host": address,
                    "port": port,
                    "family": address_family,
                    "proto": 0,
                    "flags": 0,
                }
            )
        if not resolved:
            raise OSError("host did not resolve")
        return resolved

    async def close(self) -> None:
        """Release no resources because this resolver owns no sockets."""


class PublicWebExtractor:
    """Fetch bounded HTML text from search-result URLs without credentials."""

    def __init__(
        self,
        timeout_seconds: float | None = None,
        max_bytes: int | None = None,
        max_chars: int | None = None,
        fetcher: Callable[[str], Awaitable[PublicWebResponse]] | None = None,
    ) -> None:
        """Load bounded public-page extraction settings.

        Args:
            timeout_seconds: Total timeout for one public-page request.
            max_bytes: Maximum decompressed HTML bytes accepted from one page.
            max_chars: Maximum extracted characters supplied to the model per page.
            fetcher: Test-only bounded response provider replacing network I/O.
        """
        self.timeout_seconds = _positive_float(
            timeout_seconds,
            "AI_RESEARCH_EXTRACT_TIMEOUT_SECONDS",
            12.0,
        )
        self.max_bytes = _positive_int(
            max_bytes,
            "AI_RESEARCH_EXTRACT_MAX_BYTES",
            1_048_576,
        )
        self.max_chars = _positive_int(
            max_chars,
            "AI_RESEARCH_EXTRACT_MAX_CHARS",
            6_000,
        )
        self.fetcher = fetcher

    async def extract(self, url: str) -> str | None:
        """Fetch one public HTML page and return its visible text.

        The extractor follows at most three redirects. Each redirect target is
        independently validated and the live resolver connects only to a vetted
        public IP address. Authentication, cookies, proxies from the environment,
        downloads, and non-HTML content are deliberately unavailable.

        Args:
            url: Search-provider result URL.

        Returns:
            Extracted visible text, or ``None`` when no safe public text is available.
        """
        current_url = url
        for _ in range(4):
            if not is_public_http_url(current_url):
                return None
            try:
                response = await self._fetch(current_url)
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                return None
            if len(response.body) > self.max_bytes:
                return None
            if response.status in _REDIRECT_STATUS_CODES:
                location = response.headers.get("location", "")
                if not location:
                    return None
                current_url = urljoin(current_url, location)
                continue
            if response.status != 200 or not _is_html_response(response.headers):
                return None
            return _visible_text(response.body, self.max_chars)
        return None

    async def _fetch(self, url: str) -> PublicWebResponse:
        """Request one URL without cookies, credentials, or environment proxies.

        Args:
            url: Previously validated HTTP(S) URL.

        Returns:
            Bounded response data for redirect or HTML handling.

        Raises:
            aiohttp.ClientError: If the public request fails.
            asyncio.TimeoutError: If the request exceeds its time budget.
        """
        if self.fetcher:
            return await self.fetcher(url)
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        connector = aiohttp.TCPConnector(
            resolver=_PublicDNSResolver(),
            use_dns_cache=False,
        )
        headers = {
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": "RenneBot-PublicResearch/1.0",
        }
        async with aiohttp.ClientSession(
            connector=connector,
            cookie_jar=aiohttp.DummyCookieJar(),
            headers=headers,
            timeout=timeout,
            trust_env=False,
        ) as session:
            async with session.get(url, allow_redirects=False) as response:
                content_length = response.content_length
                if content_length is not None and content_length > self.max_bytes:
                    return PublicWebResponse(response.status, response.headers, b"")
                body = bytearray()
                async for chunk in response.content.iter_chunked(16_384):
                    body.extend(chunk)
                    if len(body) > self.max_bytes:
                        return PublicWebResponse(response.status, response.headers, b"")
                return PublicWebResponse(response.status, response.headers, bytes(body))


def is_public_http_url(value: str) -> bool:
    """Reject non-web, credentialed, local, private, and unusual-port URLs.

    Args:
        value: Candidate URL supplied by a search provider or redirect.

    Returns:
        Whether the URL can proceed to DNS validation.
    """
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    if port is not None and port not in {80, 443}:
        return False
    hostname = parsed.hostname.casefold()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(
        ".local"
    ):
        return False
    try:
        return _is_public_ip(hostname)
    except ValueError:
        return True


def _is_public_ip(value: str) -> bool:
    """Return whether an IP is globally routable.

    Args:
        value: IPv4 or IPv6 address.

    Returns:
        Whether the address is globally routable and safe for public research.
    """
    return ipaddress.ip_address(value).is_global


def _is_html_response(headers: Mapping[str, str]) -> bool:
    """Allow only HTML media types.

    Args:
        headers: Response headers.

    Returns:
        Whether the response declares HTML content.
    """
    content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
    return content_type in {"text/html", "application/xhtml+xml"}


def _visible_text(body: bytes, max_chars: int) -> str | None:
    """Convert bounded HTML bytes into bounded visible text.

    Args:
        body: Bounded HTML response body.
        max_chars: Maximum characters retained for model context.

    Returns:
        Visible page text, or ``None`` when no text was found.
    """
    parser = _VisibleTextParser()
    try:
        parser.feed(body.decode("utf-8", errors="replace"))
        parser.close()
    except (LookupError, UnicodeError):
        return None
    if parser.has_password_input:
        return None
    text = "\n".join(parser.parts).strip()
    return text[:max_chars] or None


def _positive_float(value: float | None, variable: str, default: float) -> float:
    """Use a positive explicit or environment-derived floating point value.

    Args:
        value: Explicit constructor value, when provided.
        variable: Environment variable used when no explicit value is provided.
        default: Fallback for absent, invalid, or non-positive configuration.

    Returns:
        A safe positive floating point value.
    """
    if value is None:
        try:
            value = float(os.getenv(variable, str(default)))
        except ValueError:
            return default
    return value if value > 0 else default


def _positive_int(value: int | None, variable: str, default: int) -> int:
    """Use a positive explicit or environment-derived integer value.

    Args:
        value: Explicit constructor value, when provided.
        variable: Environment variable used when no explicit value is provided.
        default: Fallback for absent, invalid, or non-positive configuration.

    Returns:
        A safe positive integer value.
    """
    if value is None:
        try:
            value = int(os.getenv(variable, str(default)))
        except ValueError:
            return default
    return value if value > 0 else default
