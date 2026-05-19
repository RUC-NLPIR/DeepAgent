import os
from typing import Optional, List, Dict, Any


def exa_search(
    query: str,
    api_key: Optional[str] = None,
    num_results: int = 10,
    search_type: str = "auto",
    content_mode: str = "highlights",
    category: Optional[str] = None,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    include_text: Optional[List[str]] = None,
    exclude_text: Optional[List[str]] = None,
    start_published_date: Optional[str] = None,
    end_published_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Perform a search using the Exa API.

    Args:
        query (str): Search query.
        api_key (str): Exa API key. Falls back to EXA_API_KEY env var.
        num_results (int): Number of results to return (default 10).
        search_type (str): Search type - 'auto', 'neural', 'fast', 'instant',
                          'deep-lite', 'deep', or 'deep-reasoning'.
        content_mode (str): Content retrieval mode - 'highlights', 'text',
                           'summary', or 'none'.
        category (str): Filter by category (e.g. 'news', 'research paper',
                       'company', 'personal site', 'financial report', 'people').
        include_domains (List[str]): Only include results from these domains.
        exclude_domains (List[str]): Exclude results from these domains.
        include_text (List[str]): Strings that must appear in page text.
        exclude_text (List[str]): Strings to exclude from results.
        start_published_date (str): ISO 8601 date; only results published after this.
        end_published_date (str): ISO 8601 date; only results published before this.

    Returns:
        list: A list of dicts with keys: id, title, url, site_name, date, snippet.
              Returns empty list on failure.
    """
    api_key = api_key or os.getenv("EXA_API_KEY", "")
    if not api_key:
        print("Exa API key not found. Set EXA_API_KEY environment variable.")
        return []

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            from exa_py import Exa

            client = Exa(api_key=api_key)
            client.headers["x-exa-integration"] = "deepagent"

            search_kwargs = {
                "query": query,
                "num_results": num_results,
                "type": search_type,
            }

            # Build contents parameter
            contents = _build_contents_param(content_mode)
            if contents is not None:
                search_kwargs["contents"] = contents

            # Optional filters
            if category:
                search_kwargs["category"] = category
            if include_domains:
                search_kwargs["include_domains"] = include_domains
            if exclude_domains:
                search_kwargs["exclude_domains"] = exclude_domains
            if include_text:
                search_kwargs["include_text"] = include_text
            if exclude_text:
                search_kwargs["exclude_text"] = exclude_text
            if start_published_date:
                search_kwargs["start_published_date"] = start_published_date
            if end_published_date:
                search_kwargs["end_published_date"] = end_published_date

            response = client.search(**search_kwargs)
            return _format_results(response)

        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Exa API request failed for query: {query} after {max_retries} retries. Error: {e}")
                return []
            print(f"Exa API error occurred, retrying ({retry_count}/{max_retries})...")

    return []


async def exa_search_async(
    query: str,
    api_key: Optional[str] = None,
    num_results: int = 10,
    search_type: str = "auto",
    content_mode: str = "highlights",
    category: Optional[str] = None,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    include_text: Optional[List[str]] = None,
    exclude_text: Optional[List[str]] = None,
    start_published_date: Optional[str] = None,
    end_published_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Perform an asynchronous search using the Exa API.

    Args:
        query (str): Search query.
        api_key (str): Exa API key. Falls back to EXA_API_KEY env var.
        num_results (int): Number of results to return (default 10).
        search_type (str): Search type - 'auto', 'neural', 'fast', 'instant',
                          'deep-lite', 'deep', or 'deep-reasoning'.
        content_mode (str): Content retrieval mode - 'highlights', 'text',
                           'summary', or 'none'.
        category (str): Filter by category (e.g. 'news', 'research paper',
                       'company', 'personal site', 'financial report', 'people').
        include_domains (List[str]): Only include results from these domains.
        exclude_domains (List[str]): Exclude results from these domains.
        include_text (List[str]): Strings that must appear in page text.
        exclude_text (List[str]): Strings to exclude from results.
        start_published_date (str): ISO 8601 date; only results published after this.
        end_published_date (str): ISO 8601 date; only results published before this.

    Returns:
        list: A list of dicts with keys: id, title, url, site_name, date, snippet.
              Returns empty list on failure.
    """
    import asyncio

    api_key = api_key or os.getenv("EXA_API_KEY", "")
    if not api_key:
        print("Exa API key not found. Set EXA_API_KEY environment variable.")
        return []

    max_retries = 5
    retry_count = 0

    while retry_count < max_retries:
        try:
            from exa_py import Exa

            client = Exa(api_key=api_key)
            client.headers["x-exa-integration"] = "deepagent"

            search_kwargs = {
                "query": query,
                "num_results": num_results,
                "type": search_type,
            }

            # Build contents parameter
            contents = _build_contents_param(content_mode)
            if contents is not None:
                search_kwargs["contents"] = contents

            # Optional filters
            if category:
                search_kwargs["category"] = category
            if include_domains:
                search_kwargs["include_domains"] = include_domains
            if exclude_domains:
                search_kwargs["exclude_domains"] = exclude_domains
            if include_text:
                search_kwargs["include_text"] = include_text
            if exclude_text:
                search_kwargs["exclude_text"] = exclude_text
            if start_published_date:
                search_kwargs["start_published_date"] = start_published_date
            if end_published_date:
                search_kwargs["end_published_date"] = end_published_date

            # Run the synchronous Exa client call in an executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: client.search(**search_kwargs))
            return _format_results(response)

        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                print(f"Exa API request failed for query: {query} after {max_retries} retries. Error: {e}")
                return []
            print(f"Exa API error occurred, retrying ({retry_count}/{max_retries})...")
            await asyncio.sleep(1)

    return []


def _build_contents_param(content_mode: str) -> Optional[Dict[str, Any]]:
    """Build the contents parameter for Exa API calls based on content mode."""
    if content_mode == "highlights":
        return {"highlights": {"num_sentences": 5}}
    elif content_mode == "text":
        return {"text": {"max_characters": 10000}}
    elif content_mode == "summary":
        return {"summary": True}
    elif content_mode == "none":
        return None
    else:
        return {"highlights": {"num_sentences": 5}}


def _format_results(response) -> List[Dict[str, Any]]:
    """
    Format Exa API response into the standard result format used by DeepAgent.

    Matches the output format of extract_relevant_info_serper():
    list of dicts with keys: id, title, url, site_name, date, snippet.
    """
    results = []
    for i, result in enumerate(response.results):
        title = getattr(result, "title", "") or ""
        url = getattr(result, "url", "") or ""

        # Extract site_name from URL
        site_name = ""
        try:
            from urllib.parse import urlparse
            site_name = urlparse(url).netloc
        except Exception:
            pass

        # Extract date
        date = getattr(result, "published_date", "") or ""

        # Build snippet from content
        snippet = ""
        highlights = getattr(result, "highlights", None)
        if highlights:
            snippet = " ".join(highlights)
        else:
            text = getattr(result, "text", None)
            if text:
                snippet = text[:500]
            else:
                summary = getattr(result, "summary", None)
                if summary:
                    snippet = summary[:500]

        results.append({
            "id": i + 1,
            "title": title,
            "url": url,
            "site_name": site_name,
            "date": date.split("T")[0] if date else "",
            "snippet": snippet,
        })

    return results


def get_openai_function_exa_search() -> dict:
    """Return the OpenAI tool/function definition for exa_search."""
    return {
        "type": "function",
        "function": {
            "name": "exa_search",
            "description": (
                "Perform an AI-powered web search using Exa and return the results. "
                "Supports advanced filtering by category, domain, date range, and text content. "
                "Returns results with titles, URLs, and content snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string."
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter results by category. Options: 'news', 'research paper', 'company', 'personal site', 'financial report', 'people'.",
                        "enum": ["news", "research paper", "company", "personal site", "financial report", "people"]
                    },
                    "include_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only include results from these domains (e.g. ['arxiv.org', 'github.com'])."
                    },
                    "exclude_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exclude results from these domains."
                    },
                    "start_published_date": {
                        "type": "string",
                        "description": "Only include results published after this date (ISO 8601 format, e.g. '2024-01-01T00:00:00.000Z')."
                    },
                    "end_published_date": {
                        "type": "string",
                        "description": "Only include results published before this date (ISO 8601 format)."
                    }
                },
                "required": ["query"]
            }
        }
    }
