import os
from dotenv import load_dotenv
from backend.tools.web_tool import make_web_search_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.schema.schemas import SubAgentInput, SubAgentOutput, Source
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from pathlib import Path
import json

load_dotenv()

PATH = Path(__file__).parent.parent/"prompts"/"sub_agent.txt"
MAX_CALLS = {"shallow": 1, "deep": 3}


class SubAgent:
    """
    Stateless Research Worker.
    Receives one query, searches the web and returns the findings.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.tools = [make_web_search_tool(session_id)]
        self.llm = ChatGoogleGenerativeAI(
            model="gemma-4-26b-a4b-it",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.2
        ).bind_tools(self.tools)
        self.system_prompt = self._load_system_prompt()
        self._tool_map = {t.name: t for t in self.tools}

    def _load_system_prompt(self):
        return PATH.read_text(encoding="utf-8")

    def _build_user_message(self, inp: SubAgentInput) -> str:
        return (
            f"Parent task context: {inp.context}\n\n"
            f"Your subquery to research: {inp.subquery}\n\n"
            f"Search Depth: {inp.depth} (max {MAX_CALLS[inp.depth]} search call(s))."
        )

    def _parse_output(self, raw: str, subquery: str) -> SubAgentOutput:
        try:
            data = json.loads(raw)
            sources = [Source(**s) for s in data.get("sources", [])]
            return SubAgentOutput(
                subquery=data.get("subquery", subquery),
                status=data.get("status", "failed"),
                sources=sources,
                summary=data.get("summary", ""),
                key_facts=data.get("key_facts", []),
                confidence=float(data.get("confidence", 0.0)),
                gaps=data.get("gaps", []),
            )
        except Exception as e:
            return SubAgentOutput(
                subquery=subquery,
                status="failed",
                sources=[],
                summary=f"Failed to parse agent output: {e}\n\nRaw output: {raw}",
                key_facts=[],
                confidence=0.0,
                gaps=["Agent returned malformed JSON"],
            )

    def _execute_tool_loop(self, messages: list, max_calls: int) -> str:
        search_calls_done = 0

        while True:
            response = self.llm.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return response.content

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                if tool_name not in self._tool_map:
                    result = f"Error: unknown tool '{tool_name}'"
                else:
                    result = self._tool_map[tool_name].invoke(tool_args)
                    search_calls_done += 1

                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_id)
                )

            if search_calls_done > max_calls:
                messages.append(
                    HumanMessage(
                        content=(
                            "You have reached your search limit. "
                            "Do not call any more tools. "
                            "Now synthesize everything you found and return the final JSON output."
                        )
                    )
                )
                final_response = self.llm.invoke(messages)
                return final_response.content

    def run(self, inp: SubAgentInput) -> SubAgentOutput:
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._build_user_message(inp))
        ]

        final_text = self._execute_tool_loop(messages, max_calls=MAX_CALLS[inp.depth])
        return self._parse_output(final_text, inp.subquery)