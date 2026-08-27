# Enterprise AI Research Agent

An enterprise-grade, multi-agent research assistant that combines in-memory vector retrieval with real-time web research to produce fully cited, fact-checked intelligence reports. Powered by a deterministic LangGraph workflow, Groq LPU inference, and FAISS vector search, this application mitigates LLM hallucinations by enforcing strict verification guardrails before presenting reports to the user.

---

## Key Features

* **Multi-Agent Orchestration:** Structured graph workflow built on LangGraph, separating retrieval, synthesis, and grounding into distinct execution nodes.
* **Hybrid RAG Retrieval:** Combines local vector search over enterprise documents (via FAISS) with real-time web search (via Tavily API).
* **Automated Fact-Checking Guardrail:** Includes a verification node that cross-references draft responses against source context before final rendering.
* **Dynamic PDF Ingestion:** Upload, parse, chunk, and embed new enterprise documents directly within the UI at runtime.
* **Ultra-Fast Inference:** Utilizes Groq's LPU hardware acceleration using `openai/gpt-oss-120b` for sub-second agent node execution.
* **Transparent Audit Trail:** Displays a step-by-step reasoning panel along with clickable source citations for complete transparency.
* **Single-Canvas Streamlit Interface:** Modern, sidebar-free UI layout featuring top-level collapsible panels for settings and ingestion.

---

## Technologies Used

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit | Single-page reactive application interface |
| **Agentic Framework** | LangGraph & LangChain | State machine orchestration, tool binding, and node routing |
| **LLM Inference Engine** | Groq API (`openai/gpt-oss-120b`) | High-speed, low-latency reasoning and synthesis |
| **Vector Database** | FAISS | In-memory document similarity search |
| **Embeddings Model** | HuggingFace (`all-MiniLM-L6-v2`) | Local dense text vectorization |
| **Web Search Engine** | Tavily Search API | Real-time web search integration |
| **Document Processing** | PyPDF & LangChain Text Splitters | PDF parsing, dynamic chunking, and text extraction |

---

## Run the Deployed Application

Live Application: 

---

## How to Run the Project Second Option

1. **Clone the repository**
   ```bash
   git clone https://github.com/priyanshukushwaha0/Enterprise-AI-Research-Agent.git
   cd Enterprise-AI-Research-Agent

2. Create a virtual environment --> python -m venv myenv

3. Create a .env file -->
   
 - GROQ_API_KEY = "your_groq_api_key"
 - GROQ_MODEL = openai/gpt-oss-120b
 - TAVILY_API_KEY = "your_tavily_api_key"

5. Install dependencies --> pip install -r requirements.txt

   Run Code in a New Terminal  --> streamlit run frontend.py

---

## File Structure

    Enterprise-AI-Research-Agent/
              │
              ├── .env.example              # Template for environment variables (API keys)
              ├── requirements.txt          # Python package dependencies
              ├── config.py                 # Central configuration settings and model definitions
              ├── rag_engine.py             # FAISS vector store initialization and document ingestion
              ├── agent.py                  # LangGraph state workflow, nodes, and tool integrations
              ├── frontend.py               # Single-page Streamlit main user interface
              └── README.md                 # Project documentation

---

## Project Architecture

```text
               ┌──────────────────────────────────────────────┐
               │         Streamlit UI (frontend.py)           │
               └──────────────────────┬───────────────────────┘
                                      │ User Query & File Uploads
                                      ▼
               ┌──────────────────────────────────────────────┐
               │        LangGraph Workflow Orchestrator       │
               └──────┬───────────────────────┬───────────────┘
                      │                       │
     ┌────────────────┴──────────┐   ┌────────┴──────────────────┐
     │ 1. Local Retrieval Node   │   │ 2. Web Search Node        │
     │   (FAISS Vector Index)    │   │   (Tavily Search API)     │
     └────────────────┬──────────┘   └────────┬──────────────────┘
                      │                       │
                      └───────────┬───────────┘
                                  ▼
               ┌──────────────────────────────────────────────┐
               │      3. Report Synthesis Node (Groq LLM)     │
               └──────────────────────┬───────────────────────┘
                                      ▼
               ┌──────────────────────────────────────────────┐
               │    4. Fact-Checking & Grounding Guardrail    │
               └──────────────────────┬───────────────────────┘
                                      ▼
               ┌──────────────────────────────────────────────┐
               │    Verified Final Report & Source Citations  │
               └──────────────────────────────────────────────┘
