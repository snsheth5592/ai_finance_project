# Finance AI Project

Applied Agentic AI for SWEs - Learn By Building

## Project Structure

```
├── src/
│   ├── agents/      # AI agents
│   ├── core/        # Core logic
│   ├── data/        # Data handling
│   ├── rag/         # Retrieval-Augmented Generation
│   ├── web_app/     # Web application
│   ├── utils/       # Utilities
│   └── workflow/    # Workflow orchestration
├── tests/
├── config.yaml
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

# Finance AI Project

Applied Agentic AI for Software Engineers — Learn by Building

This project implements a multi-agent financial assistant using:
- Streamlit (UI)
- LangGraph (agent orchestration)
- LangChain + OpenAI (LLM layer)
- Pinecone (vector database for RAG)
- Yahoo Finance (market data)
- Tavily (web search)

---

# 1. Architecture Overview

## High-Level Flow

User Query
→ LangGraph Router
→ One or More Agents
→ RAG Retrieval (Pinecone)
→ LLM Processing
→ Structured Response
→ Streamlit UI

## Agents

- Finance Q&A Agent (RAG-based education)
- Portfolio Analysis Agent (detinistic + LLM summary)
- Market Analysis Agent (Yahoo Finance data)
- Goal Planning Agent (structured planning + RAG)
- News Synthesizer Agent (Yahoo + Tavily)
- Tax Education Agent (RAG-based)

## Orchestration

LangGraph controls:
- Intent routing
- Multi-agent execution
- Sequential flow when required
- State passing between agents

---

# 2. Setup Instructions

## Prerequisites

- Python 3.12 recommended
- Pinecone account
- OpenAI API key
- (Optional) Tavily API key

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables (.env)

```bash
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key
PINECONE_ENVIRONMENT=your_env
TAVILY_API_KEY=optional
```

## Run Application

```bash
streamlit run src/web_app/streamlit_app.py
```

## Testing

```bash
pytest
```

With coverage (excludes Streamlit UI for practical reasons):

```bash
pytest --cov=src --cov-report=term-missing
```

Current coverage: ~58% of core logic (agents, router, RAG, config). Streamlit app is omitted from coverage.

---

# 3. API Documentation

## Finance Q&A Agent

Input:
```json
{
  "query": "What is an ETF?"
}
```

Output:
```json
{
  "answer": "...",
  "sources": [
    {"title": "SEC Investor.gov", "url": "..."}
  ]
}
```

---

## Portfolio Agent

Input:
```json
{
  "holdings": [
    {"symbol": "AAPL", "value_usd": 10000}
  ],
  "cash_usd": 1000
}
```

Output:
```json
{
  "summary": "...",
  "metrics": {...},
  "charts": {...}
}
```

---

## Market Agent

Input:
```
"Tesla performance this week"
```

Output:
```json
{
  "symbol": "TSLA",
  "price": 123.45,
  "change_percent": 1.2
}
```

---

# 4. Usage Examples

### Example 1 – Education

User: What is diversification?

System:
- Router → Finance Q&A
- RAG → Pinecone
- LLM grounded answer

---

### Example 2 – Market + Education

User: How is Apple performing and what is an ETF?

System:
- Router → Market Agent + Finance Q&A
- Combined response

---

### Example 3 – Portfolio

User submits JSON portfolio → deterministic metrics + LLM explanation.

---

# 5. Troubleshooting Guide

## Pinecone "No records yet"
- Ensure embeddings are being upserted
- Verify dimension matches OpenAI embedding model
- Confirm namespace

## RAG returns no sources
- Verify Pinecone index exists
- Ensure ingestion ran
- Check environment variables

## Streamlit deployment errors
- Confirm environment variables configured in Streamlit Cloud
- Verify correct Python version

---

# Technical Design Document

---

## 1. System Architecture Decisions

### Why LangGraph?
- Deterministic routing
- Multi-agent composition
- Explicit state transitions
- Easier debugging than free-form agent loops

### Why Pinecone?
- Managed vector database
- Persistent cloud storage
- Scales beyond local MVP

### Why Yahoo Finance (yfinance)?
- Free
- Reliable for MVP
- No premium API constraints

---

## 2. Agent Communication Protocols

Agents communicate via structured dictionaries:

```python
{
  "answer": str,
  "sources": list,
  "metadata": dict
}
```

LangGraph state object carries:
- user_query
- chat_history
- intermediate_outputs
- selected_agents

---

## 3. RAG Implementation Details

### Embeddings
- OpenAI embeddings (1024-dim)
- Stored in Pinecone index

### Retrieval
- Top-K cosine similarity search
- Metadata contains:
  - title
  - source
  - url

### Grounded Prompting
- Retrieved chunks injected into system prompt
- LLM instructed to only answer using sources

---

## 4. Performance Considerations

### Latency
- Embedding calls are network-bound
- Pinecone query ~ few ms
- LLM response dominates total latency

### Scaling
- Pinecone handles vector scale
- Stateless Streamlit frontend
- Agents are modular

### Optimization Opportunities
- Cache embeddings
- Cache market data (5-min TTL)
- Parallel multi-agent execution in LangGraph

---

# Future Improvements

- User profiles + personalization
- Risk scoring model
- Advanced portfolio optimization
- Background memory store
- Proper observability layer

---

# License

Educational project.