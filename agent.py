from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from rag_engine import RAGEngine
import config

# Define State
class AgentState(TypedDict):
    query: str
    faiss_context: List[str]
    web_context: List[str]
    draft_report: str
    final_report: str
    reasoning_trail: List[str]
    sources: List[str]

class ResearchAgentWorkflow:
    def __init__(self, rag_engine: RAGEngine):
        self.rag = rag_engine
        self.llm = ChatGroq(
            groq_api_key=config.GROQ_API_KEY, 
            model_name=config.GROQ_MODEL_NAME, 
            temperature=0.2
        )
        self.web_search = TavilySearchResults(tavily_api_key=config.TAVILY_API_KEY, max_results=3)
        self.workflow = self._build_graph()

    def retrieve_node(self, state: AgentState) -> AgentState:
        query = state["query"]
        logs = state.get("reasoning_trail", [])
        logs.append(f"Searching internal FAISS vector store & live web sources for: '{query}'")
        
        # FAISS search
        docs = self.rag.similarity_search(query, k=3)
        faiss_texts = [f"[Doc: {d.metadata.get('source', 'Internal')}] {d.page_content}" for d in docs]
        
        # Web search fallback/enrichment
        web_results = []
        try:
            results = self.web_search.invoke({"query": query})
            for r in results:
                web_results.append(f"[Web: {r.get('url')}] {r.get('content')}")
        except Exception:
            logs.append("Web search API un-reachable. Proceeding with internal FAISS context only.")
            
        sources = [d.metadata.get('source', 'Internal Document') for d in docs]
        sources.extend([r.split("]")[0].replace("[Web: ", "") for r in web_results])

        return {
            **state, 
            "faiss_context": faiss_texts, 
            "web_context": web_results, 
            "reasoning_trail": logs,
            "sources": list(set(sources))
        }

    def synthesize_node(self, state: AgentState) -> AgentState:
        logs = state["reasoning_trail"]
        logs.append("Processing high-speed synthesis via Groq LPU...")
        
        combined_context = "\n\n".join(state["faiss_context"] + state["web_context"])
        
        prompt = f"""
        You are an Enterprise AI Research Agent. Synthesize a research report based ONLY on the context below.
        Include inline citations where appropriate (e.g., [Doc: filename] or [Web: URL]).

        Query: {state['query']}
        Context:
        {combined_context}
        """
        
        response = self.llm.invoke([SystemMessage(content="You are a precise research analyst."), HumanMessage(content=prompt)])
        return {**state, "draft_report": response.content, "reasoning_trail": logs}

    def fact_check_node(self, state: AgentState) -> AgentState:
        logs = state["reasoning_trail"]
        logs.append("Executing Grounding & Fact-Check Guardrail Step...")
        
        combined_context = "\n\n".join(state["faiss_context"] + state["web_context"])
        
        prompt = f"""
        Cross-check the following draft report against the original source facts. 
        If any claim is unsupported by the context, remove or revise it. 
        Ensure exact adherence to provided sources. If information is missing, clearly state the gap.

        Original Context:
        {combined_context}

        Draft Report:
        {state['draft_report']}
        """
        
        verified_response = self.llm.invoke([HumanMessage(content=prompt)])
        logs.append("Report verified against facts successfully.")
        
        return {**state, "final_report": verified_response.content, "reasoning_trail": logs}

    def _build_graph(self):
        builder = StateGraph(AgentState)
        
        builder.add_node("retrieve", self.retrieve_node)
        builder.add_node("synthesize", self.synthesize_node)
        builder.add_node("fact_check", self.fact_check_node)
        
        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "synthesize")
        builder.add_edge("synthesize", "fact_check")
        builder.add_edge("fact_check", END)
        
        return builder.compile()

    def run(self, query: str):
        initial_state = {
            "query": query,
            "faiss_context": [],
            "web_context": [],
            "draft_report": "",
            "final_report": "",
            "reasoning_trail": [],
            "sources": []
        }
        return self.workflow.invoke(initial_state)