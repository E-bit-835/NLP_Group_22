"""
Mobixa Mobile Shop Chatbot — Streamlit UI.

Run with:
    streamlit run src/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `python -m streamlit run src/app.py` and `streamlit run src/app.py`
# to both resolve the `src` package regardless of current working directory.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model_loader import get_resources  # noqa: E402
from src.responder import generate_response  # noqa: E402

st.set_page_config(page_title="Mobixa Chatbot", page_icon="📱", layout="centered")

SUGGESTED_QUESTIONS = [
    "How much is Redmi Note 14?",
    "Is Galaxy S24 available?",
    "Compare iPhone 15 and Galaxy S24",
    "Do you have a charger for iPhone?",
    "How much does screen replacement cost?",
    "What payment methods do you accept?",
]

WELCOME_MESSAGE = (
    "Hi, I'm the Mobixa assistant! Ask me about phone prices, stock, "
    "comparisons, accessories, repairs, or store FAQs."
)


@st.cache_resource(show_spinner="Loading Mobixa models and datasets...")
def _load_models():
    return get_resources()


def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]


def _send(user_text: str):
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.spinner("Thinking..."):
        reply = generate_response(user_text)
    st.session_state.messages.append({"role": "assistant", "content": reply})


def main():
    _load_models()  # warm/cached load, once per session
    _init_state()

    st.title("📱 Mobixa Chatbot")
    st.caption("Your mobile shop assistant — phones, accessories, repairs & more.")

    with st.sidebar:
        st.header("Suggested questions")
        for q in SUGGESTED_QUESTIONS:
            if st.button(q, use_container_width=True):
                _send(q)
                st.rerun()
        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": WELCOME_MESSAGE}
            ]
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask about a phone, accessory, repair, or FAQ...")
    if user_input:
        _send(user_input)
        st.rerun()


if __name__ == "__main__":
    main()
