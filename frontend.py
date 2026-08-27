import os
import tempfile
import streamlit as st
import config
from rag_engine import RAGEngine
from agent import ResearchAgentWorkflow

st.set_page_config(page_title="Enterprise AI Research Agent", layout="wide")

@st.cache_resource
def init_system():
    rag = RAGEngine()
    workflow = ResearchAgentWorkflow(rag)
    return rag, workflow

rag_engine, agent_workflow = init_system()

# Sidebar Setup

st.sidebar.subheader(" Dynamic Document Ingestion")
uploaded_files = st.sidebar.file_uploader("Upload internal PDF documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        rag_engine.ingest_pdf(tmp_path)
        os.remove(tmp_path)
        st.sidebar.success(f"Ingested: {uploaded_file.name}")

# Main Layout
st.title(" Enterprise AI Research Agent")

query = st.text_input("Enter your research topic or question:", placeholder="e.g., Analyze recent financial trends and internal security compliance requirements.")

if st.button("Generate Research Report", type="primary"):
    if not query.strip():
        st.warning("Please input a valid query.")
    elif not config.GROQ_API_KEY:
        st.error("Missing Groq API Key! Please set it in the sidebar or .env file.")
    else:
        with st.spinner("Agent actively working..."):
            result = agent_workflow.run(query)
            
            # Display Reasoning Trail
            with st.expander(" View Step-by-Step Reasoning Trail", expanded=False):
                for step in result["reasoning_trail"]:
                    st.write(step)
            
            # Output Display
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("### Verified Research Report")
                st.markdown(result["final_report"])
                
            with col2:
                st.markdown("### Source Citations")
                for src in result["sources"]:
                    st.markdown(f"- `{src}`")