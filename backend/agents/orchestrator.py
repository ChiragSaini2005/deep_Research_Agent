# backend/agents/orchestrator.py

from backend.tools.web_tool import make_web_search_tool
from backend.tools.sub_agent_tool import make_sub_agent_tool
from backend.tools.eval_tool import make_eval_workflow_tool, reset_eval_counter, clear_eval_counter
from backend.tools.report_writer_tool import make_report_writer_tool
from backend.utils import scratchpad
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

PATH = Path(__file__).parent.parent / "prompts" / "orchestrator.txt"


class Orchestrator:

    def __init__(self, session_id: str):
        self.session_id = session_id

        self.tools = [
            make_web_search_tool(session_id),
            make_sub_agent_tool(session_id),
            make_report_writer_tool(session_id),
            make_eval_workflow_tool(session_id),
        ]

        self.llm = ChatGoogleGenerativeAI(
            model="gemma-4-26b-a4b-it",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.2
        ).bind_tools(self.tools)

        self.system_prompt = self._load_system_prompt()
        self._tool_map = {t.name: t for t in self.tools}

    def _load_system_prompt(self) -> str:
        return PATH.read_text(encoding="utf-8")

    def _extract_text(self, response) -> str:
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                block["text"] for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return str(content)

    def _execute_tool(self, tool_call: dict) -> str:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name not in self._tool_map:
            return f"Error: unknown tool '{tool_name}'"

        result = self._tool_map[tool_name].invoke(tool_args)
        return json.dumps(result) if isinstance(result, dict) else str(result)

    def run(self, user_message: str, history: list = []) -> tuple[str, list]:
        if not history:
            scratchpad.init_scratchpad(self.session_id)
            reset_eval_counter(self.session_id)

        messages = [
            SystemMessage(content=self.system_prompt),
            *history,
            HumanMessage(content=user_message)
        ]

        while True:
            response = self.llm.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                updated_history = messages[1:]
                return self._extract_text(response), updated_history

            for tool_call in response.tool_calls:
                result = self._execute_tool(tool_call)
                messages.append(
                    ToolMessage(content=result, tool_call_id=tool_call["id"])
                )