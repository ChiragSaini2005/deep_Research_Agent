# backend/agents/title_generator.py

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

_llm = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)

_SYSTEM_PROMPT = (
    "You generate short, descriptive titles for research chat sessions. "
    "Given the user's first message, output a concise title (3-6 words). "
    "No quotes, no punctuation at the end, no markdown. "
    "Just the title text, nothing else."
)


def generate_title(user_message: str) -> str:
    try:
        response = _llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_message)
        ])

        content = response.content
        if isinstance(content, list):
            content = " ".join(
                block["text"] for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )

        title = content.strip().strip('"').strip("'")
        return title if title else "Untitled Research"

    except Exception:
        return "Untitled Research"