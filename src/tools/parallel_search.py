"""Anonymous Parallel Search MCP adapter for DeepAgent's research tools."""

import asyncio
import json
from urllib.parse import urlparse


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "DeepAgent/ParallelSearch (+https://github.com/RUC-NLPIR/DeepAgent)"
REQUEST_TIMEOUT = 60


async def _call_tool(name: str, arguments: dict) -> dict:
    # Optional dependency: the default Serper/Jina path never imports the SDK.
    try:
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:
        raise RuntimeError("Install Parallel support with pip install -r requirements-parallel.txt") from exc

    async def request():
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
        ) as client:
            async with streamable_http_client(MCP_URL, http_client=client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool(name, arguments)

    result = await asyncio.wait_for(request(), timeout=REQUEST_TIMEOUT)
    # Decode after closing the transport so tool errors retain their message
    # rather than being wrapped by its background task group.
    text = "\n".join(block.text for block in result.content if block.type == "text")
    if result.isError:
        raise RuntimeError(text or f"Parallel {name} failed")
    data = result.structuredContent
    if data is None:
        data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise RuntimeError(f"Invalid Parallel {name} response")
    if data.get("errors"):
        raise RuntimeError(f"Parallel {name} failed: {data['errors']}")
    return data


async def parallel_search(query: str, session_id: str) -> list:
    """Return DeepAgent search records, including source URLs and excerpts."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Missing required parameter: query")
    data = await _call_tool("web_search", {
        "objective": query,
        "search_queries": [query],
        "session_id": session_id,
    })
    return [
        {
            "id": index,
            "title": result.get("title") or "",
            "url": result["url"],
            "site_name": urlparse(result["url"]).netloc,
            "date": result.get("publish_date") or "",
            "snippet": "\n".join(result.get("excerpts", [])),
        }
        for index, result in enumerate(data["results"], 1)
    ]


async def parallel_fetch(urls: list, session_id: str) -> dict:
    """Return URL-to-excerpt mappings, batching the MCP's 20-URL limit."""
    if not isinstance(urls, list) or not urls:
        raise ValueError("Missing or invalid parameter: urls (non-empty list required)")
    for url in urls:
        if not isinstance(url, str) or urlparse(url).scheme not in ("http", "https") or not urlparse(url).netloc:
            raise ValueError("urls must contain HTTP or HTTPS URLs")
    urls = list(dict.fromkeys(urls))
    pages = {}
    for start in range(0, len(urls), 20):
        batch = urls[start:start + 20]
        data = await _call_tool("web_fetch", {"urls": batch, "session_id": session_id})
        for result in data["results"]:
            pages[result["url"]] = "\n".join(result.get("excerpts", []))
        if any(url not in pages or not pages[url] for url in batch):
            raise RuntimeError("Parallel web_fetch returned no content for a requested URL")
    return pages
