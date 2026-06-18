"""
LangGraph state machine wiring the CRAG + Self-RAG agentic loop (§8, P3).
"""
from __future__ import annotations

import logging
from typing import Any

from app.agents.crag import grade_documents, web_fallback
from app.agents.router import route_query
from app.agents.self_rag import ReflectionGrades, grades_pass, reflect
from app.agents.state import AgentState
from app.models import LLMProvider, Retriever

log = logging.getLogger(__name__)

_GENERATION_PROMPT = """\
You are a legal intelligence assistant. Answer the following query using ONLY the
provided context. Cite every factual claim with a source (document_id and section).
If the context is insufficient, state that explicitly.

Query: {query}

Context:
{context}

Answer (with citations):
"""


def _format_context(state: AgentState) -> str:
    return "\n---\n".join(
        f"[{d.chunk.document_type} | {d.chunk.document_id}]\n{d.chunk.content}"
        for d in state["retrieved"]
    )


def build_graph(
    retriever: Retriever,
    llm: LLMProvider,
    *,
    max_iterations: int = 3,
    crag_threshold: float = 0.7,
    tavily_api_key: str | None = None,
):
    """
    Build and return a compiled LangGraph for the agentic RAG loop.

    Nodes: route → retrieve → grade → generate → reflect → (loop or done)
    """
    try:
        from langgraph.graph import END, START, StateGraph  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("langgraph is required for the agentic loop") from exc

    def node_route(state: AgentState) -> AgentState:
        return {**state, "route": route_query(state)}

    def node_retrieve(state: AgentState) -> AgentState:
        docs = retriever.retrieve(state["query"], top_k=15)
        return {**state, "retrieved": docs}

    def node_grade(state: AgentState) -> AgentState:
        relevant, confidence = grade_documents(state["query"], state["retrieved"], llm)

        if confidence < crag_threshold and tavily_api_key:
            log.info("CRAG confidence %.2f < %.2f — triggering web fallback.", confidence, crag_threshold)
            web_docs = web_fallback(state["query"], tavily_api_key)
            relevant = relevant + web_docs
            confidence = max(confidence, 0.5)

        return {**state, "retrieved": relevant, "crag_confidence": confidence}

    def node_generate(state: AgentState) -> AgentState:
        context = _format_context(state)
        prompt = _GENERATION_PROMPT.format(query=state["query"], context=context)
        draft = llm.generate(prompt)
        return {**state, "draft": str(draft), "iterations": state.get("iterations", 0) + 1}

    def node_reflect(state: AgentState) -> AgentState:
        grades = reflect(state["query"], state["draft"], state["retrieved"], llm)
        return {
            **state,
            "reflections": {
                "is_relevant": grades.is_relevant,
                "is_faithful": grades.is_faithful,
                "is_useful": grades.is_useful,
                "retrieve_again": grades.retrieve_again,
                "reason": grades.reason,
            },
        }

    def edge_after_reflect(state: AgentState) -> str:
        iterations = state.get("iterations", 0)
        reflections = state.get("reflections", {})
        grades = ReflectionGrades(**reflections)

        if grades_pass(grades) or iterations >= max_iterations:
            if not grades_pass(grades):
                log.warning("Max iterations reached; returning best draft (low-confidence).")
            return "done"
        return "retrieve"

    def node_finalize(state: AgentState) -> AgentState:
        return {**state, "answer": state["draft"]}

    graph = StateGraph(AgentState)
    graph.add_node("route", node_route)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("grade", node_grade)
    graph.add_node("generate", node_generate)
    graph.add_node("reflect", node_reflect)
    graph.add_node("finalize", node_finalize)

    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("grade", "generate")
    graph.add_edge("generate", "reflect")
    graph.add_conditional_edges(
        "reflect",
        edge_after_reflect,
        {"retrieve": "retrieve", "done": "finalize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()
