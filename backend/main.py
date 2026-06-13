# backend/main.py
from dotenv import load_dotenv
load_dotenv()
# backend/main.py

from dotenv import load_dotenv
load_dotenv()

from backend.agents.title_generator import generate_title
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.agents.orchestrator import Orchestrator
from backend.db import database
from fastapi.responses import Response
from backend.utils.pdf_export import markdown_to_pdf
import uuid
from backend.utils import scratchpad
from backend.tools.eval_tool import clear_eval_counter

app = FastAPI(title="Deep Research Agent")

# ── In-memory runtime store ─────────────────────────────────────────────────
# holds live Orchestrator instances + LangChain message history (not persisted)
_sessions: dict[str, dict] = {}


@app.on_event("startup")
def on_startup():
    database.init_db()


# ── Request / Response Models ─────────────────────────────────────────────────

class StartRequest(BaseModel):
    message: str

class StartResponse(BaseModel):
    session_id: str
    response: str

class ReplyRequest(BaseModel):
    message: str

class ReplyResponse(BaseModel):
    response: str

class StatusResponse(BaseModel):
    session_id: str
    active: bool
    turn_count: int

class SessionSummary(BaseModel):
    id: str
    name: str
    created_at: str
    status: str

class SessionDetail(BaseModel):
    id: str
    name: str
    created_at: str
    messages: list[dict]
    status: str
    eval_score: float | None


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.post("/session/start", response_model=StartResponse)
def start_session(req: StartRequest):
    session_id = str(uuid.uuid4())

    orchestrator = Orchestrator(session_id)
    response, history = orchestrator.run(req.message, history=[])

    _sessions[session_id] = {
        "orchestrator": orchestrator,
        "history": history,
        "turn_count": 1
    }

    display_messages = [
        {"role": "user", "content": req.message},
        {"role": "assistant", "content": response}
    ]

    title = generate_title(req.message)

    database.create_session(session_id, name=title)
    database.update_session(session_id, messages=display_messages, status="active")

    return StartResponse(session_id=session_id, response=response)


@app.post("/session/{session_id}/reply", response_model=ReplyResponse)
def reply(session_id: str, req: ReplyRequest):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    response, updated_history = session["orchestrator"].run(
        req.message,
        history=session["history"]
    )

    session["history"] = updated_history
    session["turn_count"] += 1

    # update persisted display messages
    db_session = database.get_session(session_id)
    display_messages = db_session["messages"]
    display_messages.append({"role": "user", "content": req.message})
    display_messages.append({"role": "assistant", "content": response})
    database.update_session(session_id, messages=display_messages)

    return ReplyResponse(response=response)


@app.get("/session/{session_id}/status", response_model=StatusResponse)
def get_status(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    return StatusResponse(
        session_id=session_id,
        active=True,
        turn_count=session["turn_count"]
    )


@app.get("/sessions", response_model=list[SessionSummary])
def list_sessions():
    return database.list_sessions()


@app.get("/session/{session_id}", response_model=SessionDetail)
def get_session_detail(session_id: str):
    session = database.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session




@app.delete("/session/{session_id}")
def end_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]

    scratchpad.clear(session_id)
    clear_eval_counter(session_id)
    database.delete_session(session_id)
    return {"detail": "Session deleted"}



@app.get("/session/{session_id}/pdf")
def export_pdf(session_id: str):
    session = database.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = session["messages"]
    if not messages:
        raise HTTPException(status_code=400, detail="No content to export")

    # original query = first user message
    query = next((m["content"] for m in messages if m["role"] == "user"), "")
    # final answer = last assistant message
    final_answer = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")

    pdf_bytes = markdown_to_pdf(
        title=session["name"],
        query=query,
        body_markdown=final_answer
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{session["name"]}.pdf"'}
    )