# Mini RAG — Haystack over .tex papers (Groq-backed)

A compact, single-file Retrieval-Augmented-Generation (RAG) demo built with Haystack
that ingests LaTeX papers, chunks and embeds them, stores them in an in-memory Haystack
document store (persisted to disk), then answers queries using the Groq API.

This repository contains a single runnable script: `rag_pipeline.py` (ingest → index → query).

**Quick overview**
- Ingest: parse `.tex` files in `data/raw/papers/` → `data/processed/raw_docs.json`
- Index: split, embed, and save a Haystack `InMemoryDocumentStore` at `data/processed/docstore.pkl`
- Query: retrieve top-k chunks and generate an answer with Groq (auto-selects an available model)

Prerequisites
- Python 3.10+ (works with homebrew / system Python on macOS)
- A Groq API key (set `GROQ_API_KEY` in your environment). Optionally a `GOOGLE_API_KEY` if you prefer Google Gemini.

One-command setup
```bash
python -m venv .venv && source .venv/bin/activate && .venv/bin/python -m pip install -U pip && .venv/bin/python -m pip install -r requirements.txt
```

Basic file layout
- `rag_pipeline.py` — single script to ingest/index/query
- `data/raw/papers/` — put your `.tex` paper files here
- `data/processed/raw_docs.json` — plain-text parsed docs (created by `--ingest`)
- `data/processed/docstore.pkl` — persisted Haystack in-memory store (created by `--ingest`)

Usage

- Ingest LaTeX papers
```bash
.venv/bin/python rag_pipeline.py --ingest
```

- List available Groq models (does not run a query)
```bash
GROQ_API_KEY="<your_key>" .venv/bin/python rag_pipeline.py --groq-models
```

- Run a query (preferred: use `GROQ_API_KEY`)
```bash
GROQ_API_KEY="<your_key>" .venv/bin/python rag_pipeline.py --query "Your question here"
```

**Chat modes**

Two chat modes are available:

1. **Standard chat (`--chat`)**: Papers-only mode. The assistant answers strictly based on the provided paper context and will say when the papers don't cover a topic.

```bash
GROQ_API_KEY="<your_key>" .venv/bin/python rag_pipeline.py --chat
```

2. **Enhanced chat (`--chat-enhanced`)**: Papers + external knowledge mode. The assistant uses the papers as the primary source but may draw on general knowledge to elaborate, connect ideas, or fill gaps. It clearly marks when stepping beyond the papers (e.g., "Based on general knowledge...").

```bash
GROQ_API_KEY="<your_key>" .venv/bin/python rag_pipeline.py --chat-enhanced
```

Both modes support follow-up questions within the same session. Use `/exit` or `/quit` to leave.

Example session

1. Input: list the available Groq models.
```bash
GROQ_API_KEY="<your_key>" .venv/bin/python rag_pipeline.py --groq-models
```

Output:
```text
Found 16 models:
- llama-3.1-8b-instant | active=True | owner=Meta | context=131072 | max_tokens=131072
- groq/compound | active=True | owner=Groq | context=131072 | max_tokens=8192
- groq/compound-mini | active=True | owner=Groq | context=131072 | max_tokens=8192
- llama-3.3-70b-versatile | active=True | owner=Meta | context=131072 | max_tokens=32768
- openai/gpt-oss-20b | active=True | owner=OpenAI | context=131072 | max_tokens=65536
```

2. Input: ask the RAG system a question.
```bash
GROQ_API_KEY="<your_key>" .venv/bin/python rag_pipeline.py --query "what is LAS function class?"
```

Output:
```text
Loading document store from /path/to/rag-eval/data/processed/docstore.pkl
Warming up text embedder...
Retrieved 5 chunks

Question: what is LAS function class?

Answer:
The LAS function class (short for Locally Asymptotically Symmetric) consists of
functions for which the Bregman asymmetry near a minimizer is negligible compared
with the quadratic term. In practice, it includes strongly convex, smooth functions
and some C^1 functions with well-behaved local curvature.

Sources used:
	- HNAGppJOTA.tex (chunk 21)
	- HNAGppJOTA.tex (chunk 22)
	- HNAGppJOTA.tex (chunk 23)
	- HNAGppJOTA.tex (chunk 25)
	- HNAGppJOTA.tex (chunk 28)
```

Environment variables
- `GROQ_API_KEY` — required to use Groq for generation
- `GROQ_MODEL` — optional: suggested model id (e.g. `groq/compound-mini`); otherwise the script will query Groq for active models and try them in order
- `GOOGLE_API_KEY` — optional: if present and `GROQ_API_KEY` absent the script will use Google Gemini (via haystack_integrations)
- `HF_TOKEN` — optional: used by `sentence-transformers` to speed model downloads from HF

Behavior notes
- The pipeline uses `SentenceTransformers` (`BAAI/bge-small-en-v1.5`) for embeddings.
- The `--groq-models` flag prints active models (ID, owner, context window, max tokens).
- When `GROQ_API_KEY` is present the script prefers Groq. It will try `GROQ_MODEL` (if set), then fetch the live list of active models from Groq and try each until one returns a successful completion.
- **Chat mode (`--chat`)**: Strictly paper-focused. The assistant will only use provided context and says clearly when papers don't cover a topic. Ideal for fact-checking or extracting information strictly from the papers.
- **Enhanced chat mode (`--chat-enhanced`)**: Papers + general knowledge. The assistant can elaborate beyond the papers while clearly marking external contributions (e.g., "Based on general knowledge..."). Ideal for exploratory discussions, connecting ideas, or understanding broader context.

Troubleshooting
- "document store not found": run `--ingest` first to create `data/processed/docstore.pkl`.
- "model_decommissioned" or other model errors: run `--groq-models` to see currently active models, then set `GROQ_MODEL` to a supported id and retry.
- If embedding downloads are slow or rate-limited, set `HF_TOKEN` or pre-download models manually.

Advanced ideas
- Add `--prefer-cheap` (not implemented) to bias selection toward `mini`/`compound-mini` models.
- Save answers to `data/processed/last_answer.json` by piping script output or extend `rag_pipeline.py` to persist responses.

License & safety
- This demo is intended for local experimentation. Keep API keys private and avoid committing them to source control.

If you want, I can:
- Add a `requirements.txt` and a Makefile for one-line setup, or
- Implement a `--prefer-cheap` flag to bias model selection toward cheaper models.

Enjoy — ask me to run a query or tweak the prompts if you'd like different behavior.
