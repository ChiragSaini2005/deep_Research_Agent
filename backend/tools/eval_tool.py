from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from backend.agents.eval_workflow import _build_eval_workflow
from backend.schema.schemas import EvalFeedback

MAX_RUNS = 2
_run_counts: dict[str, int] = {}


def make_eval_workflow_tool(session_id: str):

    _eval_workflow = _build_eval_workflow(session_id)

    @tool
    def EvalWorkflowTool(original_query: str, consolidated_research: str) -> dict:
        """
        Evaluates the consolidated research answer for factual consistency and writing quality.
        Returns structured feedback including overall score, topics needing more research,
        and sections that need rewriting.
        IMPORTANT: This tool can only be called a maximum of 2 times per research task.
        If score >= 0.80, the research has passed. Otherwise use the feedback to improve.
        """
        current_count = _run_counts.get(session_id, 0)

        if current_count >= MAX_RUNS:
            return EvalFeedback(
                overall_score=0.0,
                passed=False,
                needs_more_research=[],
                sections_to_rewrite=[],
                notes=["Evaluation limit reached. Maximum 2 evaluations allowed per research task."]
            ).model_dump()

        _run_counts[session_id] = current_count + 1

        result = _eval_workflow.invoke({
            "original_query": original_query,
            "consolidated_research": consolidated_research
        })

        return result["feedback"].model_dump()

    return EvalWorkflowTool


def reset_eval_counter(session_id: str):
    _run_counts[session_id] = 0


def clear_eval_counter(session_id: str):
    _run_counts.pop(session_id, None)