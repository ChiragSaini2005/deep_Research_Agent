from pydantic import BaseModel, Field
from typing import Literal


#SubAgent Schema
class SubAgentInput(BaseModel):
    subquery: str = Field(..., description= "The specific research question agent must answer")
    context: str = Field(..., description= "Brief summary of the parent task for relevance grounding")
    depth: Literal["shallow", "deep"] = Field(default = "shallow", description= "shallow = 1 search pass, deep = upto 3 search passes")

class Source(BaseModel):
    url: str
    title: str
    snippet: str
    relevance_score: float = Field(..., ge = 0.0, le = 1.0)

class SubAgentOutput(BaseModel):
    subquery: str
    status: Literal["succes", "partial", "failed"]
    sources: list[Source]
    summary: str
    key_facts: list[str]
    confidence: float = Field(..., ge = 0.0, le = 1.0)
    gaps: list[str]


#Evaluation Workflow Schema
class SectionToRewrite(BaseModel):
    section: str        # which part of the answer needs rewriting
    reason: str         # why it needs rewriting


class EvalFeedback(BaseModel):
    overall_score: float                        # 0.0 – 1.0
    passed: bool                                # score >= 0.80
    needs_more_research: list[str]              # topics that need more research
    sections_to_rewrite: list[SectionToRewrite]
    notes: list[str] 