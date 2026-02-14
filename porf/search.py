"""Search engines."""

from abc import ABC, abstractmethod
from typing import Optional
import httpx

from .types import Source


class SearchEngine(ABC):
    """Base search engine interface."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[Source]:
        """Search by query."""
        pass


class TavilySearch(SearchEngine):
    """Tavily - best for research."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "include_raw_content": False,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        return [
            Source(
                url=r["url"],
                title=r.get("title", ""),
                content=r.get("content", ""),
                query=query,
            )
            for r in data.get("results", [])
        ]


class BraveSearch(SearchEngine):
    """Brave Search API."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={"X-Subscription-Token": self.api_key},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        return [
            Source(
                url=r["url"],
                title=r.get("title", ""),
                content=r.get("description", ""),
                query=query,
            )
            for r in data.get("web", {}).get("results", [])
        ]


class SerperSearch(SearchEngine):
    """Serper.dev - Google Search API."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        resp = httpx.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": max_results},
            headers={"X-API-KEY": self.api_key},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        return [
            Source(
                url=r["link"],
                title=r.get("title", ""),
                content=r.get("snippet", ""),
                query=query,
            )
            for r in data.get("organic", [])[:max_results]
        ]


class DuckDuckGoSearch(SearchEngine):
    """DuckDuckGo - free, no API key required."""

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        try:
            from ddgs import DDGS
        except ImportError:
            raise ImportError("pip install ddgs")

        results = []
        ddgs = DDGS()
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                Source(
                    url=r["href"],
                    title=r.get("title", ""),
                    content=r.get("body", ""),
                    query=query,
                )
            )
        return results


class SearxNGSearch(SearchEngine):
    """SearxNG - self-hosted meta-search engine."""

    def __init__(self, instance_url: str):
        self.url = instance_url.rstrip('/')

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        resp = httpx.get(
            f"{self.url}/search",
            params={"q": query, "format": "json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        return [
            Source(
                url=r["url"],
                title=r.get("title", ""),
                content=r.get("content", ""),
                query=query,
            )
            for r in data.get("results", [])[:max_results]
        ]


def get_search_engine(name: str, api_key: Optional[str] = None) -> SearchEngine:
    """Search engine factory.

    For SearxNG, pass instance URL as api_key or set SEARXNG_URL env var.
    """
    import os

    if name == "searxng":
        url = api_key or os.getenv("SEARXNG_URL")
        if not url:
            raise ValueError("SearxNG URL required. Set SEARXNG_URL or pass api_key= with the instance URL")
        return SearxNGSearch(url)

    engines = {
        "tavily": (TavilySearch, "TAVILY_API_KEY"),
        "brave": (BraveSearch, "BRAVE_API_KEY"),
        "serper": (SerperSearch, "SERPER_API_KEY"),
        "duckduckgo": (DuckDuckGoSearch, None),
    }

    if name not in engines:
        available = list(engines) + ["searxng"]
        raise ValueError(f"Unknown search engine: {name}. Available: {available}")

    cls, env_var = engines[name]

    if env_var:
        key = api_key or os.getenv(env_var)
        if not key:
            raise ValueError(f"API key required. Set {env_var} or pass api_key=")
        return cls(key)

    return cls()
