# deep_research_agent_round_scoped.py
# ------------------------------------------------------------
# Deep Research Agent (LangGraph 0.6.7)
# - Parallel search/read via Command(goto=[Send(...), ...])
# - True fan-in barriers using round-scoped marker lists (no counter bleed-through)
# - Clean iteration handoff via next_subqueries
# - Delta-only node returns; reducers for parallel merges
# ------------------------------------------------------------

from __future__ import annotations

import os, re, time, json
from dataclasses import dataclass
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from operator import add

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.types import Send, Command
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

import httpx
import trafilatura
from pydantic import BaseModel

# Optional search providers
TAVILY_AVAILABLE = False
try:
    from tavily import TavilyClient
    if os.getenv("TAVILY_API_KEY"):
        TAVILY_AVAILABLE = True
except Exception:
    TAVILY_AVAILABLE = False


# =======================
# LLM
# =======================

def llm(temperature: float = 0.2):
    model = os.getenv("MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=temperature)


# =======================
# State
# =======================

class Doc(TypedDict):
    url: str
    title: str
    summary: str
    content: str

class SearchHit(TypedDict):
    query: str
    url: str
    title: str
    snippet: str

class ResearchState(TypedDict):
    # core
    question: str
    plan: str

    # iteration query ownership
    subqueries: List[str]           # LastValue – only subqueries_node writes per loop
    next_subqueries: List[str]      # LastValue – reflect parks follow-ups here

    # aggregates (parallel)
    searches: Annotated[List[SearchHit], add]
    docs: Annotated[List[Doc], add]
    notes: Annotated[List[str], add]

    # outputs
    answer_draft: str
    citations: List[str]

    # loop control
    iteration: int
    max_iterations: int
    done: bool

    # barriers (round-scoped)
    search_round: int               # LastValue – current search fan-out round id
    expected_search: int            # LastValue – workers we expect for this round
    search_marks: Annotated[List[int], add]  # reducer – each worker appends round id on completion

    read_round: int                 # LastValue – current read fan-out round id
    expected_read: int              # LastValue – workers we expect for this round
    read_marks: Annotated[List[int], add]    # reducer – each worker appends round id on completion


# =======================
# Pydantic Outputs
# =======================

class PlanOutput(BaseModel):
    plan: str

class SubqueriesOutput(BaseModel):
    subqueries: List[str]

class DocSummaryOutput(BaseModel):
    summary: str

class SynthesizeOutput(BaseModel):
    answer_draft: str

class ReflectionOutput(BaseModel):
    gaps: List[str]
    followups: List[str]
    decision: str
    why: str


# =======================
# Utils & Web
# =======================

def clean_text(txt: str, max_chars: int = 40_000) -> str:
    import re as _re
    txt = _re.sub(r"\s+", " ", txt).strip()
    return txt[:max_chars]

def web_search(query: str, k: int = 6) -> List[SearchHit]:
    hits: List[SearchHit] = []
    if TAVILY_AVAILABLE:
        try:
            tv = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            res = tv.search(query=query, max_results=k)
            items = res.get("results", []) if isinstance(res, dict) else (res if isinstance(res, list) else [])
            for r in items:
                if isinstance(r, dict):
                    url = r.get("url") or ""
                    title = r.get("title") or url
                    snippet = (r.get("content") or r.get("snippet") or "")[:300]
                elif isinstance(r, str):
                    url = r if r.startswith(("http://", "https://")) else ""
                    title, snippet = (url or "Result"), ""
                else:
                    continue
                if url:
                    hits.append({"query": query, "url": url, "title": title, "snippet": snippet})
            return hits[:k]
        except Exception:
            pass
    
    return [{"query": query, "url": "", "title": "No search backend configured", "snippet": "Install tavily or ddg."}]

def web_fetch(url: str, timeout: float = 25.0) -> Optional[Doc]:
    if not url or not url.startswith(("http://", "https://")):
        return None
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent":"DeepResearchAgent/1.0"}) as client:
            r = client.get(url); r.raise_for_status()
            downloaded = trafilatura.extract(r.text, include_comments=False, include_images=False, url=url)
            content = clean_text(downloaded if downloaded else r.text)
            import re as _re
            title_m = _re.search(r"<title>(.*?)</title>", r.text, _re.I)
            title = title_m.group(1).strip() if title_m else url
            return {"url": url, "title": title, "summary": "", "content": content}
    except Exception:
        return None


# =======================
# Prompts
# =======================

SYSTEM_BRIEF = """You are a meticulous Deep Research Agent.
You iterate: PLAN → SUB-QUERIES → SEARCH → READ → SYNTHESIZE → REFLECT → DECIDE (continue/stop).
Always track sources (URLs) and avoid hallucinations.
Prefer authoritative, recent, and diverse sources.
When evidence is weak, say so.
"""

PROMPT_PLAN = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BRIEF),
    ("human", """User question:
{question}

Draft a brief plan (2–5 bullet points) for how you will research and answer this. Focus on sub-questions and source types.
Return strict JSON:
{{
  "plan": "A short paragraph or bullet points as a single string."
}}
""")
])

PROMPT_SUBQUERIES = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BRIEF),
    ("human", """Question:
{question}

Plan:
{plan}

Propose 3–5 concrete search sub-queries (no duplicates). Return strict JSON:
{{
  "subqueries": ["...", "..."]
}}
""")
])

PROMPT_DOC_SUMMARY = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BRIEF),
    ("human", """Summarize the following web page into 5–8 crisp bullet points geared toward answering:
{question}

Title: {title}
URL: {url}
Content (truncated):
{content}

Return strict JSON:
{{
  "summary": "A single string containing bullet points, not a list."
}}
""")
])

PROMPT_SYNTHESIZE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BRIEF),
    ("human", """You have the user's question, notes, and summaries from multiple sources.

Question:
{question}

Notes so far:
{notes}

Source digests (JSON list of {{url, title, key_points}}):
{digests}

Write a concise, well-structured answer (<= 400 words) with inline numeric citations like [1], [2] that map to a References list of URLs at the end. Be balanced about uncertainty. Only cite sources that support claims.
Return strict JSON:
{{
  "answer_draft": "..."
}}
""")
])

PROMPT_REFLECT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BRIEF),
    ("human", """Reflect on the current draft answer **specifically with respect to the user's question**.

User Question:
{question}

Current Draft Answer:
{answer_draft}

Identify:
1) Missing perspectives or contradictions worth checking.
2) At most 3 targeted follow-up sub-queries.
3) Verdict: "continue" or "stop" with a one-sentence justification.

Return strict JSON:
{{
  "gaps": ["..."],
  "followups": ["..."],
  "decision": "continue|stop",
  "why": "..."
}}
""")
])


# =======================
# Nodes
# =======================

def node_plan(state: ResearchState) -> Dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=PlanOutput)
    chain = PROMPT_PLAN | llm() | parser
    result = chain.invoke({"question": state["question"]})
    return {"plan": result.plan, "notes": [f"Plan:\n{result.plan}"]}

def node_subqueries(state: ResearchState):
    # consume follow-ups if present; else generate fresh
    if state.get("next_subqueries"):
        subqs = [q.strip() for q in state["next_subqueries"] if q.strip()]
    else:
        parser = PydanticOutputParser(pydantic_object=SubqueriesOutput)
        chain = PROMPT_SUBQUERIES | llm() | parser
        result = chain.invoke({"question": state["question"], "plan": state["plan"]})
        seen, subqs = set(), []
        for q in result.subqueries:
            q = q.strip()
            if q and q not in seen:
                seen.add(q); subqs.append(q)
            if len(subqs) >= 5:
                break

    # new search round id
    next_round = state.get("search_round", 0) + 1
    sends = [Send("search_worker", {"q": q, "round": next_round}) for q in subqs]

    return {
        "subqueries": subqs,
        "next_subqueries": [],
        "search_round": next_round,
        "expected_search": len(sends),
        # NOTE: we do NOT clear search_marks (reducer list); routers count only current round
        # Optional: clear aggregates per iter:
        # "searches": [], "docs": [],
    }, Command(goto=sends)

def search_worker(arg: Dict[str, Any]) -> Dict[str, Any]:
    subquery, round_id = arg["q"], arg["round"]
    hits = web_search(subquery, k=5)
    seen, results = set(), []
    for h in hits:
        url = h.get("url") if isinstance(h, dict) else None
        if url and url not in seen:
            results.append(h); seen.add(url)
    # mark completion by appending the round id
    return {"searches": results, "search_marks": [round_id]}

def search_join(state: ResearchState) -> Dict[str, Any]:
    # dedupe + cap
    dedup, seen = [], set()
    for h in state["searches"]:
        u = h.get("url")
        if u and u not in seen:
            seen.add(u); dedup.append(h)
    # debug
    current = state.get("search_round", 0)
    count = sum(1 for rid in state.get("search_marks", []) if rid == current)
    print(f"[search_join] round={current} marks={count}/{state.get('expected_search',0)}")
    return {"searches": dedup[:10]}

def plan_reads(state: ResearchState):
    top = state["searches"][:10]
    next_round = state.get("read_round", 0) + 1
    sends = [Send("read_worker", {"url": h["url"], "question": state["question"], "round": next_round}) for h in top]
    return {
        "read_round": next_round,
        "expected_read": len(sends),
    }, Command(goto=sends)

def read_worker(arg: Dict[str, Any]) -> Dict[str, Any]:
    url, question, round_id = arg["url"], arg["question"], arg["round"]
    doc = web_fetch(url)
    if not doc or not doc.get("content"):
        return {"docs": [], "read_marks": [round_id]}
    parser = PydanticOutputParser(pydantic_object=DocSummaryOutput)
    chain = PROMPT_DOC_SUMMARY | llm(temperature=0.1) | parser
    summary_obj = chain.invoke({
        "question": question,
        "title": doc["title"],
        "url": doc["url"],
        "content": doc["content"][:12000],
    })
    doc["summary"] = summary_obj.summary
    time.sleep(0.15)
    return {"docs": [doc], "read_marks": [round_id]}

def read_join(state: ResearchState) -> Dict[str, Any]:
    current = state.get("read_round", 0)
    count = sum(1 for rid in state.get("read_marks", []) if rid == current)
    print(f"[read_join] round={current} marks={count}/{state.get('expected_read',0)}")
    return {"docs": state["docs"][:5]}

def node_synthesize(state: ResearchState) -> Dict[str, Any]:
    digests = []
    for d in state["docs"]:
        bullets = [b.strip("-• ").strip() for b in re.split(r"[\r\n]+", d["summary"]) if b.strip()]
        digests.append({"url": d["url"], "title": d["title"], "key_points": bullets[:10]})
    parser = PydanticOutputParser(pydantic_object=SynthesizeOutput)
    chain = PROMPT_SYNTHESIZE | llm(temperature=0.2) | parser
    result = chain.invoke({
        "question": state["question"],
        "notes": "\n".join(state["notes"])[:4000],
        "digests": json.dumps(digests, ensure_ascii=False)[:15000],
    })
    urls_in_order: List[str] = []
    for d in state["docs"]:
        if d["url"] not in urls_in_order:
            urls_in_order.append(d["url"])
    return {"citations": urls_in_order, "answer_draft": result.answer_draft}

def node_reflect(state: ResearchState) -> Dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=ReflectionOutput)
    chain = PROMPT_REFLECT | llm(temperature=0.2) | parser
    reflection = chain.invoke({"question": state["question"], "answer_draft": state["answer_draft"]})

    new_notes = []
    if reflection.gaps:
        new_notes.append("Gaps identified:\n- " + "\n- ".join(reflection.gaps[:5]))
    if reflection.followups:
        new_notes.append("Follow-up subqueries:\n- " + "\n- ".join(reflection.followups[:3]))

    next_iter = state["iteration"] + 1
    wants_continue = reflection.decision.lower() == "continue"
    under_cap = next_iter < state["max_iterations"]
    should_continue = wants_continue and under_cap

    delta: Dict[str, Any] = {"iteration": next_iter, "done": not should_continue}
    if new_notes:
        delta["notes"] = new_notes
    if reflection.followups:
        delta["next_subqueries"] = reflection.followups[:3]

    print(f"[reflect] iter={state['iteration']} -> next={next_iter}, decision={reflection.decision}, done={not should_continue}")
    return delta


# =======================
# Barrier Routers
# =======================

def can_advance_after_search(state: ResearchState) -> str:
    expected = state.get("expected_search", 0)
    current_round = state.get("search_round", 0)
    marks = state.get("search_marks", [])
    done = sum(1 for rid in marks if rid == current_round)
    # Defensive: at zero expected, advance immediately; clamp otherwise.
    return "go" if (expected == 0 or done >= expected) else "wait"

def can_advance_after_read(state: ResearchState) -> str:
    expected = state.get("expected_read", 0)
    current_round = state.get("read_round", 0)
    marks = state.get("read_marks", [])
    done = sum(1 for rid in marks if rid == current_round)
    return "go" if (expected == 0 or done >= expected) else "wait"

def should_continue(state: ResearchState) -> str:
    return "stop" if (state["done"] or state["iteration"] >= state["max_iterations"]) else "loop"


# =======================
# Build Graph
# =======================

def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("plan_node", node_plan)
    graph.add_node("subqueries_node", node_subqueries)

    graph.add_node("search_worker", search_worker)
    graph.add_node("search_join", search_join)

    graph.add_node("plan_reads", plan_reads)
    graph.add_node("read_worker", read_worker)
    graph.add_node("read_join", read_join)

    graph.add_node("synthesize", node_synthesize)
    graph.add_node("reflect", node_reflect)

    graph.set_entry_point("plan_node")
    graph.add_edge("plan_node", "subqueries_node")

    # search fan-out/fan-in
    graph.add_edge("subqueries_node", "search_join")
    graph.add_edge("search_worker", "search_join")
    graph.add_conditional_edges("search_join", can_advance_after_search, {
        "go": "plan_reads",
        "wait": "search_join",  # self-loop until marks for current round reach expected
    })

    # read fan-out/fan-in
    graph.add_edge("plan_reads", "read_join")
    graph.add_edge("read_worker", "read_join")
    graph.add_conditional_edges("read_join", can_advance_after_read, {
        "go": "synthesize",
        "wait": "read_join",
    })

    graph.add_edge("synthesize", "reflect")
    graph.add_conditional_edges("reflect", should_continue, {
        "loop": "subqueries_node",
        "stop": END,
    })

    return graph.compile(checkpointer=MemorySaver())


# =======================
# Runner
# =======================

@dataclass
class RunConfig:
    max_iterations: int = 2

def run_deep_research(question: str, cfg: RunConfig = RunConfig()) -> Dict[str, Any]:
    app = build_graph()
    state: ResearchState = {
        "question": question,
        "plan": "",
        "subqueries": [],
        "next_subqueries": [],
        "searches": [],
        "docs": [],
        "notes": [],
        "answer_draft": "",
        "citations": [],
        "iteration": 0,
        "max_iterations": cfg.max_iterations,
        "done": False,
        # rounds & barriers
        "search_round": 0,
        "expected_search": 0,
        "search_marks": [],
        "read_round": 0,
        "expected_read": 0,
        "read_marks": [],
    }

    config = {"configurable": {"thread_id": "deep-research-session-1"}}

    # Optional: visualize
    # print(app.get_graph().draw_ascii())

    while True:
        state = app.invoke(state, config=config)
        if state["done"]:
            break

    final = (state["answer_draft"] or "").rstrip()
    if "References" not in final and state["citations"]:
        refs = "\n".join([f"[{i+1}] {u}" for i, u in enumerate(state["citations"])])
        final += "\n\nReferences\n" + refs

    return {
        "answer": final,
        "iterations": state["iteration"],
        "plan": state["plan"],
        "sources": state["citations"][:20],
    }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]).strip() or "What is Agentic AI? Give a concise overview with citations."
    out = run_deep_research(q, RunConfig(max_iterations=2))
    print("\n" + "="*80)
    print("FINAL ANSWER\n")
    print(out["answer"])
    print("\n" + "="*80)
    print("PLAN\n")
    print(out["plan"])
    print("\nSOURCES")
    for s in out["sources"]:
        print("-", s)
