"""SafeChat-AI Streamlit interface.

Run with:  streamlit run app.py
"""
import streamlit as st

from chatbot import process_message

st.set_page_config(page_title="SafeChat-AI", page_icon="\U0001F916")
st.title("SafeChat-AI \U0001F916")
st.caption("An AI-Powered Harmful Speech Detection and Moderation Chatbot")

USER_ID = "demo_user"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg.get("meta"):
            with st.expander("Moderation details"):
                col1, col2 = st.columns(2)
                meta = msg["meta"]
                with col1:
                    st.metric("Classification", meta.get("severity_label", "N/A").capitalize())
                    st.metric("Model Confidence", f"{meta.get('model_confidence', 0):.2%}")
                with col2:
                    st.metric("Severity Score", f"{meta.get('severity_score', 0):.2f}")
                    st.metric("Keyword Hits", meta.get("keyword_hits", 0))
                if meta.get("suggested_alternative"):
                    st.info(f"Suggested alternative: {meta['suggested_alternative']}")
                st.json(meta)

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "text": user_input, "meta": None})
    with st.chat_message("user"):
        st.markdown(user_input)

    history = [
        {"role": m["role"], "text": m["text"]} for m in st.session_state.messages[:-1]
    ]
    result = process_message(USER_ID, user_input, history=history)

    icon = {"blocked_message": "\u26A0\uFE0F", "blocked_user": "\U0001F6AB", "chat": "\U0001F916"}.get(
        result["type"], "\U0001F916"
    )

    with st.chat_message("assistant"):
        st.markdown(f"{icon} {result['reply']}")
        if result.get("analysis"):
            with st.expander("Moderation details"):
                meta = result["analysis"]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Classification", meta.get("severity_label", "N/A").capitalize())
                    st.metric("Model Confidence", f"{meta.get('model_confidence', 0):.2%}")
                with col2:
                    st.metric("Severity Score", f"{meta.get('severity_score', 0):.2f}")
                    st.metric("Keyword Hits", meta.get("keyword_hits", 0))
                if meta.get("suggested_alternative"):
                    st.info(f"Suggested alternative: {meta['suggested_alternative']}")
                st.json(meta)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "text": f"{icon} {result['reply']}",
            "meta": result.get("analysis"),
        }
    )

with st.sidebar:
    st.subheader("About SafeChat-AI")
    st.write(
        "Every message passes through a **two-layer architecture**:"
    )
    st.markdown("---")
    st.markdown(
        "**1. Safety & Moderation Layer**\n\n"
        "- BERT-based harmful speech detection\n"
        "- Severity classification (safe / mild / moderate / severe)\n"
        "- Message blocking & violation tracking\n"
        "- Respectful alternative suggestions"
    )
    st.markdown("---")
    st.markdown(
        "**2. Conversation Layer**\n\n"
        "- Intent-based response generation\n"
        "- Response filtering through moderation model\n"
        "- Maintains conversation context"
    )
    st.markdown("---")
    st.caption("Built for university CSE project & GitHub portfolio.")