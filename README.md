# Deep Research Agent

A LangGraph-powered multi-step research agent that goes beyond basic RAG (retrieve-and-generate).
Instead of just “search → stuff into context → answer,” this agent performs parallel searches, synchronized reading, iterative synthesis, and reflection loops to deliver more accurate and well-rounded research answers.

## ✨ Features

### Plan → Subqueries → Search → Read → Synthesize → Reflect → Iterate

### Parallel search and read workers (fan-out / fan-in pattern)

### Synchronization barriers to ensure all searches and reads finish before moving forward

### Delta dict updates (only diffs flow through state → memory-efficient)

### Reflection loop: the agent critiques its own draft, proposes follow-ups, and improves answers

### Safe by design: ignores hallucinations and always cites URLs

## 🛠️ Tech Stack

LangGraph  0.6.7
LangChain Core
LangChain OpenAI
OpenAI Models (default: gpt-4o-mini)
Trafilatura (webpage text extraction)
Tavily (search backend)
httpx (fetching with timeouts and redirects)

## 📦 Installation

Clone the repo:
    git clone https://github.com/nagulapavan/deep_Research_Agent.git
    cd deep_Research_Agent

Create a virtual environment:
    python -m venv venv
    source venv/bin/activate   # Mac/Linux
    venv\Scripts\activate      # Windows

Install dependencies:
    pip install -r requirements.txt

## 🔑 Environment Setup

Create a .env file in the root directory:

OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here   # optional
MODEL=gpt-4o-mini

## 🚀 Usage

Run the agent from command line:

    python deep_research_agent_round_scoped.py "What is Agentic AI? Summarize with citations."

Example output:
    ================================================================================
    FINAL ANSWER

    Agentic AI is an emerging field where LLMs act as autonomous decision-making
    agents rather than passive responders. It combines planning, memory,
    and tool-use to achieve goals in dynamic environments... [1][2]

    References
    [1] https://arxiv.org/abs/2401....
    [2] https://towardsdatascience.com/...
    ================================================================================
    PLAN

    - Define Agentic AI
    - Identify academic and industry examples
    - Compare with traditional LLM usage
    - Assess challenges and future directions

    SOURCES
    - https://arxiv.org/abs/2401....
    - https://towardsdatascience.com/...

## 📊 How It Works
The workflow looks like this:

![High Level Graph](<Untitled diagram _ Mermaid Chart-2025-09-15-160824.png>)


## 🔄 Key Concepts

### Fan-out / Fan-in: Multiple searches or reads run in parallel, then results are merged.
### Barriers: search_join and read_join wait until all workers for that round finish.
### Delta dicts: Nodes only return changes (deltas), which reducers merge into state.
### Reflection: After synthesis, the agent critiques its own work and may trigger another iteration.

## 📌 Roadmap

 ### Add Gaurdrails using LLM-based URL and content safety checks
 ### Add vector store for memory across runs
 ### Add GUI (Streamlit/Gradio) for interactive deep research

## 📜 License

MIT License.
