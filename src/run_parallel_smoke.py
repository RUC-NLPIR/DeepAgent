"""Exercise native research tool dispatch without starting a reasoning model."""

import argparse
import asyncio
import json
from types import SimpleNamespace

from tools.tool_manager import ToolManager


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', default='Python asyncio official documentation')
    args = parser.parse_args()
    manager = await ToolManager.create(SimpleNamespace(
        dataset_name='gaia', enable_tool_search=False, research_web_provider='parallel',
    ))
    seq = {}
    results = await manager.call_tool({"function": {
        "name": "web_search", "arguments": {"query": args.query},
    }}, seq)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if not isinstance(results, list) or not results:
        raise RuntimeError("Parallel search did not return any results")
    pages = await manager.call_tool({"function": {
        "name": "browse_pages", "arguments": {"urls": [results[0]['url']]},
    }}, seq)
    print(json.dumps(pages, indent=2, ensure_ascii=False))
    if 'error' in pages or not pages.get(results[0]['url']):
        raise RuntimeError("Parallel fetch did not return page excerpts")


if __name__ == '__main__':
    asyncio.run(main())
