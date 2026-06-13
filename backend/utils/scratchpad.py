# backend/utils/scratchpad.py

_scratchpads: dict[str, list[dict]] = {}


def init_scratchpad(session_id: str):
    _scratchpads[session_id] = []


def store(session_id: str, entry: dict):
    if session_id not in _scratchpads:
        _scratchpads[session_id] = []
    _scratchpads[session_id].append(entry)


def get_all(session_id: str) -> list[dict]:
    return _scratchpads.get(session_id, [])


def clear(session_id: str):
    _scratchpads.pop(session_id, None)