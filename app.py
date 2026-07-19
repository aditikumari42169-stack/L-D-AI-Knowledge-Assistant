import os
import streamlit as st

from rag import build_index, ask_ai

def load_css():
    with open("styles/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="L&D AI Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

load_css()

st.markdown("""
<div class="hero">

<h1>📚 L&D AI Knowledge Assistant</h1>

<p>
Your Intelligent Learning Companion
</p>

<span>
Search SOPs • Policies • Training • FAQs
</span>

</div>
""", unsafe_allow_html=True)

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

