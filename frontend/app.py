# frontend/app.py

import streamlit as st
import httpx

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔬",
    layout="wide"
)

# ── Session State Init ────────────────────────────────────────────────────────

defaults = {
    "session_id": None,
    "messages": [],
    "turn_count": 0,
    "eval_score": None,
    "status": "idle",
    "confirm_delete": None,   # session_id pending delete confirmation
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── API Helpers ───────────────────────────────────────────────────────────────

def start_session(message: str) -> str:
    res = httpx.post(f"{BACKEND_URL}/session/start", json={"message": message}, timeout=3000.0)
    res.raise_for_status()
    data = res.json()
    st.session_state.session_id = data["session_id"]
    st.session_state.turn_count = 1
    return data["response"]


def send_reply(message: str) -> str:
    res = httpx.post(
        f"{BACKEND_URL}/session/{st.session_state.session_id}/reply",
        json={"message": message}, timeout=300.0
    )
    res.raise_for_status()
    data = res.json()
    st.session_state.turn_count += 1
    return data["response"]


def fetch_sessions() -> list[dict]:
    try:
        res = httpx.get(f"{BACKEND_URL}/sessions", timeout=10.0)
        res.raise_for_status()
        return res.json()
    except Exception:
        return []


def load_session(session_id: str):
    res = httpx.get(f"{BACKEND_URL}/session/{session_id}", timeout=10.0)
    res.raise_for_status()
    data = res.json()

    st.session_state.session_id = session_id
    st.session_state.messages = data["messages"]
    st.session_state.turn_count = len(data["messages"]) // 2
    st.session_state.eval_score = data.get("eval_score")
    st.session_state.status = "done" if data["messages"] else "idle"


def delete_session_api(session_id: str):
    httpx.delete(f"{BACKEND_URL}/session/{session_id}")
    if st.session_state.session_id == session_id:
        new_chat()


def new_chat():
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.turn_count = 0
    st.session_state.eval_score = None
    st.session_state.status = "idle"


def handle_send(user_input: str):
    if not user_input.strip():
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.status = "researching"

    try:
        if st.session_state.session_id is None:
            response = start_session(user_input)
        else:
            response = send_reply(user_input)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.status = "done"

    except httpx.HTTPStatusError as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error: {e.response.status_code} — {e.response.text}"
        })
        st.session_state.status = "failed"
    except httpx.TimeoutException:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Error: Request timed out. The research may still be running on the server."
        })
        st.session_state.status = "failed"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Research Agent")

    if st.button("➕ New Research", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()

    tab_history, tab_progress = st.tabs(["History", "Progress"])

    # ── History Tab ──
    with tab_history:
        sessions = fetch_sessions()

        if not sessions:
            st.caption("No past sessions yet.")

        for sess in sessions:
            is_active = sess["id"] == st.session_state.session_id

            row = st.container()
            with row:
                col_name, col_menu = st.columns([5, 1])

                with col_name:
                    label = f"**{sess['name']}**" if is_active else sess["name"]
                    if st.button(label, key=f"load_{sess['id']}", use_container_width=True):
                        load_session(sess["id"])
                        st.rerun()

                with col_menu:
                    with st.popover("⋮", use_container_width=True):
                        if st.button("🗑️ Delete", key=f"del_{sess['id']}"):
                            st.session_state.confirm_delete = sess["id"]
                            st.rerun()

            # confirmation prompt
            if st.session_state.confirm_delete == sess["id"]:
                st.warning(f"Delete '{sess['name']}'?")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Yes", key=f"confirm_yes_{sess['id']}", use_container_width=True):
                        delete_session_api(sess["id"])
                        st.session_state.confirm_delete = None
                        st.rerun()
                with c2:
                    if st.button("No", key=f"confirm_no_{sess['id']}", use_container_width=True):
                        st.session_state.confirm_delete = None
                        st.rerun()

    # ── Progress Tab ──
    with tab_progress:
        status_map = {
            "idle":        ("⏳", "Waiting for query..."),
            "researching": ("🔍", "Researching..."),
            "done":        ("✅", "Done"),
            "failed":      ("❌", "Failed"),
        }
        icon, label = status_map[st.session_state.status]
        st.markdown(f"**Status:** {icon} {label}")

        st.divider()
        st.metric("Turns", st.session_state.turn_count)

        if st.session_state.eval_score is not None:
            st.metric(
                "Eval Score",
                f"{int(st.session_state.eval_score * 100)}%",
                delta="Passed ✅" if st.session_state.eval_score >= 0.8 else "Failed ❌"
            )

        if st.session_state.status == "done" and st.session_state.session_id:
            st.divider()
            try:
                pdf_res = httpx.get(
                    f"{BACKEND_URL}/session/{st.session_state.session_id}/pdf",
                    timeout=30.0
                )
                if pdf_res.status_code == 200:
                    st.download_button(
                        "📄 Download as PDF",
                        data=pdf_res.content,
                        file_name="research_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            except Exception:
                pass


# ── Main Chat Window ─────────────────────────────────────────────────────────

st.header("🔬 Deep Research Agent")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.status == "researching":
    with st.chat_message("assistant"):
        with st.spinner("Researching... this may take a while"):
            st.empty()

user_input = st.chat_input("Ask me anything...")
if user_input:
    handle_send(user_input)
    st.rerun()