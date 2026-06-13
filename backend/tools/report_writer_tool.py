# backend/tools/report_writer_tool.py

import os
import json
from pathlib import Path
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from backend.utils import scratchpad

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "report_writer.txt"

_llm = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)


def _extract_text(response) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def make_report_writer_tool(session_id: str):

    @tool
    def ReportWriterTool(original_query: str) -> str:
        """
        Writes a comprehensive, long-form research report based on ALL findings
        gathered so far (from sub-agents and web searches). Call this when you
        have finished researching and are ready to produce the final report.

        Args:
            original_query: The user's original research question
        """
        entries = scratchpad.get_all(session_id)

        if not entries:
            return "Error: No research findings available yet. Research before calling this tool."

        system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
        findings_text = "\n\n---\n\n".join(
            json.dumps(entry, indent=2) for entry in entries
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Original query: {original_query}\n\n"
                f"Research findings:\n{findings_text}"
            ))
        ]

        response = _llm.invoke(messages)
        report = _extract_text(response)

        # store the report itself in the scratchpad too, so re-runs (after eval feedback)
        # have access to the previous draft
        scratchpad.store(session_id, {
            "type": "report_draft",
            "data": report
        })

        return report

    return ReportWriterTool