from langchain_core.tools import tool
from backend.agents.sub_agent import SubAgent
from backend.schema.schemas import SubAgentInput
from backend.utils import scratchpad



def make_sub_agent_tool(session_id: str):

    _sub_agent = SubAgent(session_id)  # shared, stateless — safe to reuse across sessions
    
    @tool
    def SubAgentTool(subquery: str, context: str, depth: str = "shallow") -> str:
        """
        A research sub-agent that searches the web to answer a specific subquery.
        Use this tool when you need to delegate a focused research task.

        Args:
            subquery: The specific research question to answer
            context: Brief summary of the parent task for relevance grounding
            depth: 'shallow' for 1 search pass, 'deep' for up to 3 passes
        """
        inp = SubAgentInput(subquery=subquery, context=context, depth=depth)
        result = _sub_agent.run(inp)

        # full result goes to scratchpad
        scratchpad.store(session_id, {
            "type": "sub_agent_result",
            "subquery": subquery,
            "data": result.model_dump()
        })

        # orchestrator only sees a short summary
        return (
            f"[{result.status}] {result.summary[:200]}"
            f"{'...' if len(result.summary) > 200 else ''} "
            f"(confidence: {result.confidence})"
        )

    return SubAgentTool