# Deep Research Agent

An autonomous multi-agent research system that takes a user query, clarifies ambiguity, decomposes it into subqueries, researches in parallel using sub-agents and web search, writes a comprehensive long-form report, and self-evaluates the report before delivering it.

## Architecture

```
User
 │
 ▼
Orchestrator (Gemma 4 26B A4B IT)
 │
 ├── PHASE 1: Clarify (asks up to 3 clarifying questions)
 ├── PHASE 2: Research
 │     ├── WebSearchTool (Tavily)        — quick lookups
 │     └── SubAgentTool                  — delegated deep research
 │           └── SubAgent (own WebSearchTool)
 ├── PHASE 3: Write Report
 │     └── ReportWriterTool              — long-form markdown report
 │           (reads full research scratchpad)
 └── PHASE 4: Evaluate
       └── EvalWorkflowTool (LangGraph)  — max 2 calls per session
             ├── Factual Consistency Eval (with WebSearchTool)
             └── Writing Quality Eval
                   └── Feedback Synthesis → score, gaps, sections to rewrite
```

If the evaluation score is below 80%, the orchestrator re-researches weak areas and regenerates the report (up to 2 evaluation cycles).

## Features

- **Conversational clarification** — asks up to 3 clarifying questions before researching
- **Multi-agent research** — orchestrator delegates complex subqueries to sub-agents, each with their own web search
- **Long-form reports** — dedicated Report Writer produces structured, in-depth markdown reports from a research scratchpad (avoids context overflow)
- **Self-evaluation loop** — LangGraph-based workflow checks factual consistency and writing quality, with up to 2 retry cycles
- **Session history** — all sessions persisted to SQLite, with LLM-generated titles
- **PDF export** — download any completed report as a styled PDF
- **LangSmith observability** — full tracing of every LLM call and tool invocation

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| Frontend | Streamlit |
| LLM | Google Gemma (via Gemini API) |
| Agent Framework | LangChain, LangGraph |
| Web Search | Tavily |
| Database | SQLite |
| PDF Generation | ReportLab |
| Observability | LangSmith |

## Project Structure

```
deep_research_agent/
├── backend/
│   ├── main.py                    # FastAPI app & endpoints
│   ├── agents/
│   │   ├── orchestrator.py        # Main orchestrator agent
│   │   ├── sub_agent.py           # Research sub-agent
│   │   ├── eval_nodes.py          # LangGraph eval workflow nodes
│   │   └── title_generator.py     # Session title generation
│   ├── tools/
│   │   ├── web_tool.py            # Web search tool (factory)
│   │   ├── sub_agent_tool.py      # Sub-agent as a tool (factory)
│   │   ├── eval_tool.py           # Eval workflow as a tool (factory)
│   │   └── report_writer_tool.py  # Report writer tool (factory)
│   ├── schema/
│   │   ├── schemas.py             # Core Pydantic models
│   │   └── eval_schemas.py        # Evaluation workflow schemas
│   ├── utils/
│   │   ├── scratchpad.py          # Per-session research scratchpad
│   │   └── pdf_export.py          # Markdown → PDF conversion
│   ├── db/
│   │   └── database.py            # SQLite session persistence
│   └── prompts/
│       ├── orchestrator.txt
│       ├── sub_agent.txt
│       ├── report_writer.txt
│       ├── factual_eval.txt
│       ├── writing_eval.txt
│       └── feedback_gen.txt
├── frontend/
│   └── app.py                     # Streamlit UI
├── requirements.txt
└── .env                           # API keys (not committed)
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/deep-research-agent.git
cd deep-research-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key

# LangSmith (optional, for observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=deep-research-agent
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google AI Studio API key for Gemma models |
| `TAVILY_API_KEY` | Yes | Tavily Search API key |
| `LANGCHAIN_TRACING_V2` | No | Set to `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | LangSmith project name |
| `LANGCHAIN_ENDPOINT` | No | LangSmith endpoint URL |

### 5. Run the backend

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`.

### 6. Run the frontend

In a new terminal (with the virtual environment activated):

```bash
streamlit run frontend/app.py
```

The UI will open at `http://localhost:8501`.

## Usage

1. Type a research query into the chat
2. Answer any clarifying questions the agent asks (up to 3)
3. Watch the **Progress** tab in the sidebar for live status and evaluation score
4. Once complete, view the full report in the chat and download it as a PDF
5. Past sessions appear in the **History** tab — click to view, or use the `⋮` menu to delete

## How It Works

### Research Scratchpad

To support long-form reports without overflowing the LLM context window, all raw research findings (sub-agent results, web search results) are stored in a per-session scratchpad rather than the orchestrator's conversation history. The orchestrator only sees compact summaries. The **Report Writer** reads the full scratchpad in a dedicated call to produce the final report.

### Evaluation Loop

After the report is generated, `EvalWorkflowTool` runs a LangGraph workflow with two parallel evaluators:

- **Factual Consistency** — checks for contradictions and unsupported claims (has its own web search access to verify facts)
- **Writing Quality** — checks completeness, structure, and clarity

Both feed into a feedback synthesis node that produces an overall score (0–1) and specific notes on what needs more research or rewriting. If the score is below 0.80, the orchestrator uses this feedback to improve the report. This can happen at most twice per session.

## Limitations (Iteration 1)

- Sessions are held in memory for live conversation continuation; restarting the backend means past sessions can be viewed (from SQLite) but not continued
- Research scratchpad and eval counters are in-memory only, cleared on session deletion or backend restart
- PDF export uses a minimal markdown parser (headings, bullets, bold/italic) — tables and nested lists are not supported
