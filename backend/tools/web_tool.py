# backend/tools/web_tool.py

from langchain_core.tools import tool
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from backend.utils import scratchpad

load_dotenv()

_tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def make_web_search_tool(session_id: str):

    @tool
    def WebSearchTool(query: str) -> str:
        """A search engine optimized for AI agents. Use this tool to search the live web
        for real-time information, current events, news, and facts. Input should be a
        concise search query (under 400 characters)."""
        response = _tavily_client.search(query)
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item["title"],
                "url": item["url"],
                "snippet": item["content"]
            })

        # full results go to scratchpad
        scratchpad.store(session_id, {
            "type": "web_search_result",
            "query": query,
            "data": results
        })

        # orchestrator sees a compact summary
        summary_lines = [f"- {r['title']}: {r['snippet'][:100]}..." for r in results[:5]]
        return f"Found {len(results)} results:\n" + "\n".join(summary_lines)

    return WebSearchTool