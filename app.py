import os
import streamlit as st

from rag import build_index, ask_ai

st.set_page_config(
    page_title="L&D AI Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 L&D AI Knowledge Assistant")
st.write("Ask questions from the organization's knowledge base.")

# Build Index Button
if st.button("Build Knowledge Base"):
    with st.spinner("Creating FAISS index..."):
        build_index()
    st.success("Knowledge Base Built Successfully!")

st.divider()

# Question Box
question = st.text_input(
    "Ask your question:",
    placeholder="Example: How do I request training?"
)

# Ask Button
if st.button("Ask AI"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = ask_ai(question)

        st.subheader("Answer")
        st.write(answer)
