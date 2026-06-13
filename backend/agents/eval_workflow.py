import os
import json
from pathlib import Path
from pydantic import BaseModel
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from backend.tools.web_tool import make_web_search_tool
from backend.schema.schemas import SectionToRewrite, EvalFeedback
from langgraph.graph import StateGraph, START, END


class EvalWorkflowState(TypedDict):
    original_query: str
    consolidated_research: str

    factual_score: float
    factual_issues: list[str]
    writing_score: float
    writing_issues: list[str]

    feedback: EvalFeedback


PROMPTS = Path(__file__).parent.parent / "prompts"
MODEL = "gemma-4-31b-it"


def _get_llm(with_tools_list=None):
    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1
    )
    return llm.bind_tools(with_tools_list) if with_tools_list else llm


def make_factual_eval_node(session_id: str):
    web_search_tool = make_web_search_tool(session_id)
    tool_map = {web_search_tool.name: web_search_tool}

    def factual_eval_node(state: EvalWorkflowState) -> dict:
        system_prompt = (PROMPTS / "factual_eval.txt").read_text(encoding="utf-8")
        llm = _get_llm(with_tools_list=[web_search_tool])

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Original query: {state['original_query']}\n\n"
                f"Consolidated research:\n{state['consolidated_research']}"
            ))
        ]

        while True:
            response = llm.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                tool = tool_map.get(tool_call["name"])
                result = tool.invoke(tool_call["args"]) if tool else "Unknown tool"
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

        try:
            data = json.loads(response.content)
            return {
                "factual_score": float(data.get("factual_score", 0.0)),
                "factual_issues": data.get("factual_issues", [])
            }
        except Exception as e:
            return {
                "factual_score": 0.0,
                "factual_issues": [f"Factual eval failed to parse: {e}"]
            }

    return factual_eval_node


def writing_eval_node(state: EvalWorkflowState) -> dict:
    system_prompt = (PROMPTS / "writing_eval.txt").read_text(encoding="utf-8")
    llm = _get_llm()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Original query: {state['original_query']}\n\n"
            f"Consolidated research:\n{state['consolidated_research']}"
        ))
    ]

    response = llm.invoke(messages)

    try:
        data = json.loads(response.content)
        return {
            "writing_score": float(data.get("writing_score", 0.0)),
            "writing_issues": data.get("writing_issues", [])
        }
    except Exception as e:
        return {
            "writing_score": 0.0,
            "writing_issues": [f"Writing eval failed to parse: {e}"]
        }


def feedback_gen_node(state: EvalWorkflowState) -> dict:
    system_prompt = (PROMPTS / "feedback_gen.txt").read_text(encoding="utf-8")
    llm = _get_llm()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Original query: {state['original_query']}\n\n"
            f"Factual score: {state['factual_score']}\n"
            f"Factual issues: {json.dumps(state['factual_issues'])}\n\n"
            f"Writing score: {state['writing_score']}\n"
            f"Writing issues: {json.dumps(state['writing_issues'])}"
        ))
    ]

    response = llm.invoke(messages)

    try:
        data = json.loads(response.content)
        feedback = EvalFeedback(
            overall_score=float(data.get("overall_score", 0.0)),
            passed=bool(data.get("passed", False)),
            needs_more_research=data.get("needs_more_research", []),
            sections_to_rewrite=[
                SectionToRewrite(**s) for s in data.get("sections_to_rewrite", [])
            ],
            notes=data.get("notes", [])
        )
        return {"feedback": feedback}
    except Exception as e:
        return {
            "feedback": EvalFeedback(
                overall_score=0.0,
                passed=False,
                needs_more_research=[],
                sections_to_rewrite=[],
                notes=[f"Feedback generation failed to parse: {e}"]
            )
        }


def _build_eval_workflow(session_id: str):
    graph = StateGraph(EvalWorkflowState)

    graph.add_node("factual_eval", make_factual_eval_node(session_id))
    graph.add_node("writing_eval", writing_eval_node)
    graph.add_node("feedback_gen", feedback_gen_node)

    graph.add_edge(START, "factual_eval")
    graph.add_edge(START, "writing_eval")
    graph.add_edge("factual_eval", "feedback_gen")
    graph.add_edge("writing_eval", "feedback_gen")
    graph.add_edge("feedback_gen", END)

    return graph.compile()