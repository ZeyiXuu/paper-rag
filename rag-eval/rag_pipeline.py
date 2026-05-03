"""Single-file scaffold for the mini Haystack RAG pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from pylatexenc.latex2text import LatexNodes2Text


MATH_ENV_PATTERN = re.compile(
    r"\\begin\{(?:equation\*?|align\*?|gather\*?|displaymath)\}.*?"
    r"\\end\{(?:equation\*?|align\*?|gather\*?|displaymath)\}",
    re.DOTALL,
)
STRIP_COMMAND_PATTERN = re.compile(
    r"\\(?:cite|ref|label)\s*(?:\[[^\]]*\])?\s*\{[^}]*\}",
)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line parser.

    Returns:
        argparse.ArgumentParser: Parser with ingest and query options.
    """
    parser = argparse.ArgumentParser(description="Mini RAG pipeline scaffold")
    parser.add_argument("--ingest", action="store_true", help="Parse .tex files and build the index")
    parser.add_argument("--query", type=str, help="Query the saved document store")
    parser.add_argument("--chat", action="store_true", help="Start a chat session (papers only mode)")
    parser.add_argument("--chat-enhanced", action="store_true", help="Start an enhanced chat session (papers + external knowledge)")
    parser.add_argument("--groq-models", action="store_true", help="List available Groq models and exit")
    return parser


def load_document_store(repo_root: Path) -> InMemoryDocumentStore | None:
    """Load the persisted in-memory document store from disk."""
    store_path = repo_root / "data" / "processed" / "docstore.pkl"
    if not store_path.exists():
        print(f"Error: document store not found at {store_path}. Run --ingest first.")
        return None

    print(f"Loading document store from {store_path}")
    return InMemoryDocumentStore.load_from_disk(str(store_path))


def build_context(store: InMemoryDocumentStore, query: str, top_k: int = 5) -> tuple[list[str], list[str]]:
    """Retrieve top-k chunks and format them for prompting."""
    text_embedder = SentenceTransformersTextEmbedder(model="BAAI/bge-small-en-v1.5")
    print("Warming up text embedder...")
    text_embedder.warm_up()
    query_embedding = text_embedder.run(text=query)["embedding"]

    retriever = InMemoryEmbeddingRetriever(document_store=store, top_k=top_k)
    retrieved_docs = retriever.run(query_embedding=query_embedding)["documents"]
    print(f"Retrieved {len(retrieved_docs)} chunks")

    context_blocks: list[str] = []
    source_lines: list[str] = []
    for doc in retrieved_docs:
        source = str(doc.meta.get("source", "unknown_source"))
        split_id = int(doc.meta.get("split_id", 0)) + 1
        context_blocks.append(f"[{source} chunk {split_id}]\n{doc.content}")
        source_lines.append(f"  - {source} (chunk {split_id})")

    return context_blocks, source_lines


def generate_answer(messages: list[dict[str, str]], groq_key: str | None, google_key: str | None) -> str:
    """Generate an answer from a chat message list using Groq or Gemini."""
    if groq_key:
        try:
            import groq as _groq

            client = _groq.Groq(api_key=groq_key)

            # Build candidate model list: env override first, then live active models from API
            candidates: list[str] = []
            env_model = os.environ.get("GROQ_MODEL")
            if env_model:
                candidates.append(env_model)
            try:
                model_list_resp = client.models.list()
                ml = model_list_resp.model_dump() if hasattr(model_list_resp, "model_dump") else getattr(model_list_resp, "__dict__", {})
                for entry in ml.get("data", []) if isinstance(ml, dict) else []:
                    if entry.get("active"):
                        candidates.append(entry.get("id"))
            except Exception:
                # If listing fails, we'll still try the env model (if any) and fall back later
                pass

            # Deduplicate while preserving order
            seen = set()
            candidates = [m for m in candidates if m and not (m in seen or seen.add(m))]

            resp = None
            last_exc = None
            for model_name in candidates or [os.environ.get("GROQ_MODEL", "llama-3.2-3b-preview")]:
                try:
                    resp = client.chat.completions.create(messages=messages, model=model_name)
                    break
                except Exception as e:
                    last_exc = e

            if resp is None:
                raise last_exc or RuntimeError("No Groq model produced a successful completion")

            def _extract_text(obj) -> str:
                # Try pydantic model dumping first
                try:
                    data = obj.model_dump()  # type: ignore[attr-defined]
                except Exception:
                    try:
                        data = obj.__dict__
                    except Exception:
                        data = str(obj)

                # Search common fields for text
                def walk(d):
                    if d is None:
                        return None
                    if isinstance(d, str):
                        return d
                    if isinstance(d, dict):
                        for k in ("text", "content", "message", "output", "messages", "choices"):
                            if k in d:
                                v = d[k]
                                if isinstance(v, list):
                                    for item in v:
                                        t = walk(item)
                                        if t:
                                            return t
                                else:
                                    t = walk(v)
                                    if t:
                                        return t
                        # fallback deeper search
                        for v in d.values():
                            t = walk(v)
                            if t:
                                return t
                    if isinstance(d, list):
                        for item in d:
                            t = walk(item)
                            if t:
                                return t
                    return None

                found = walk(data)
                return found or ""

            return _extract_text(resp)
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            return f"Groq generation failed: {exc}"

    if not google_key:
        return "Error: neither GROQ_API_KEY nor GOOGLE_API_KEY is set. Export one before running --query."

    from haystack_integrations.components.generators.google_ai import GoogleAIGeminiChatGenerator

    generator = GoogleAIGeminiChatGenerator(model="gemini-2.0-flash-lite")
    chat_messages = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            chat_messages.append(ChatMessage.from_system(content))
        elif role == "assistant":
            chat_messages.append(ChatMessage.from_assistant(content))
        else:
            chat_messages.append(ChatMessage.from_user(content))

    replies = generator.run(messages=chat_messages)["replies"]
    return replies[0].text if replies else "No answer returned by Gemini."


def clean_latex(text: str) -> str:
    """Convert LaTeX source to plain text with minimal cleanup.

    Args:
        text: Raw LaTeX source text.

    Returns:
        Clean plain-text content.
    """
    def _unwrap_math_env(match: re.Match[str]) -> str:
        matched = match.group(0)
        for wrapper in (
            (r"\\begin\{equation\*?\}", r"\\end\{equation\*?\}"),
            (r"\\begin\{align\*?\}", r"\\end\{align\*?\}"),
            (r"\\begin\{gather\*?\}", r"\\end\{gather\*?\}"),
            (r"\\begin\{displaymath\}", r"\\end\{displaymath\}"),
        ):
            inner = re.sub(wrapper[0], "", matched, count=1)
            inner = re.sub(wrapper[1], "", inner, count=1)
            return inner.strip()
        return matched

    text = MATH_ENV_PATTERN.sub(_unwrap_math_env, text)
    text = STRIP_COMMAND_PATTERN.sub("", text)
    plain_text = LatexNodes2Text().latex_to_text(text)
    plain_text = re.sub(r"\n{3,}", "\n\n", plain_text)
    plain_text = re.sub(r"[ \t]+", " ", plain_text)
    return plain_text.strip()


def ingest_papers(repo_root: Path) -> list[dict[str, str]]:
    """Parse all .tex papers and write extracted plain text JSON.

    Args:
        repo_root: Root directory that contains data/raw/papers.

    Returns:
        List of parsed documents with source metadata.
    """
    raw_dir = repo_root / "data" / "raw" / "papers"
    processed_dir = repo_root / "data" / "processed"
    output_path = processed_dir / "raw_docs.json"

    processed_dir.mkdir(parents=True, exist_ok=True)
    tex_files = sorted(raw_dir.glob("*.tex"))
    print(f"Found {len(tex_files)} .tex files in {raw_dir}")

    docs: list[dict[str, str]] = []
    for tex_file in tex_files:
        raw_text = tex_file.read_text(encoding="utf-8", errors="ignore")
        clean_text = clean_latex(raw_text)
        docs.append({"source": tex_file.name, "text": clean_text})
        print(f"Parsed {tex_file.name} ({len(clean_text)} chars)")

    output_path.write_text(json.dumps(docs, indent=2), encoding="utf-8")
    print(f"Saved {len(docs)} documents to {output_path}")
    return docs


def build_index(repo_root: Path, docs: list[dict[str, str]]) -> Path:
    """Split, embed, and persist papers into an in-memory Haystack store.

    Args:
        repo_root: Root directory containing data/processed output files.
        docs: Parsed plain-text paper documents.

    Returns:
        Path to the serialized document store.
    """
    processed_dir = repo_root / "data" / "processed"
    store_path = processed_dir / "docstore.pkl"

    source_docs = [Document(content=item["text"], meta={"source": item["source"]}) for item in docs]
    print(f"Prepared {len(source_docs)} documents for splitting")

    splitter = DocumentSplitter(split_by="word", split_length=400, split_overlap=20)
    split_result = splitter.run(documents=source_docs)
    split_docs = split_result["documents"]
    print(f"Split into {len(split_docs)} chunks")

    embedder = SentenceTransformersDocumentEmbedder(model="BAAI/bge-small-en-v1.5")
    print("Warming up document embedder...")
    embedder.warm_up()
    embedded_result = embedder.run(documents=split_docs)
    embedded_docs = embedded_result["documents"]
    print(f"Embedded {len(embedded_docs)} chunks")

    store = InMemoryDocumentStore()
    written_count = store.write_documents(embedded_docs)
    print(f"Wrote {written_count} embedded chunks to InMemoryDocumentStore")

    store.save_to_disk(str(store_path))
    print(f"Saved document store to {store_path}")
    return store_path


def query_papers(repo_root: Path, query: str) -> None:
    """Run retrieval and Gemini generation for a user query.

    Args:
        repo_root: Root directory containing the processed document store.
        query: User question text.
    """
    # Prefer GROQ if available in env, otherwise fall back to Google Gemini
    groq_key = os.environ.get("GROQ_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")

    store = load_document_store(repo_root)
    if store is None:
        return

    context_blocks, source_lines = build_context(store, query)

    system_prompt = (
        "Answer the question using only the provided context from research papers. "
        "Cite the source filename for each claim. If the context doesn't contain "
        "the answer, say so clearly."
    )
    user_prompt = (
        f"Question:\n{query}\n\n"
        "Context:\n"
        + "\n\n".join(context_blocks)
    )
    answer = generate_answer(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        groq_key,
        google_key,
    )

    print(f"Question: {query}\n")
    print("Answer:")
    print(answer)
    print("\nSources used:")
    for line in source_lines:
        print(line)


def chat_papers(repo_root: Path) -> None:
    """Run an interactive RAG chat session that keeps prior turns in memory."""
    groq_key = os.environ.get("GROQ_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")

    store = load_document_store(repo_root)
    if store is None:
        return

    system_prompt = (
        "Answer the question using only the provided context from research papers. "
        "Use the conversation history when the user asks a follow-up question. "
        "Cite the source filename for each claim. If the context doesn't contain "
        "the answer, say so clearly."
    )
    conversation_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    print("Chat mode ready. Type a question and press Enter. Type /exit or /quit to stop.")
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat mode.")
            return

        if not query:
            continue
        if query.lower() in {"/exit", "/quit"}:
            print("Exiting chat mode.")
            return

        context_blocks, source_lines = build_context(store, query)
        user_prompt = (
            f"Question:\n{query}\n\n"
            "Context:\n"
            + "\n\n".join(context_blocks)
        )
        answer = generate_answer(conversation_messages + [{"role": "user", "content": user_prompt}], groq_key, google_key)

        print("\nAssistant:")
        print(answer)
        print("\nSources used:")
        for line in source_lines:
            print(line)

        conversation_messages.append({"role": "user", "content": query})
        conversation_messages.append({"role": "assistant", "content": answer})


def chat_papers_enhanced(repo_root: Path) -> None:
    """Run an interactive RAG chat session that allows external knowledge while citing papers."""
    groq_key = os.environ.get("GROQ_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")

    store = load_document_store(repo_root)
    if store is None:
        return

    system_prompt = (
        "You are a knowledgeable research assistant. Use the provided context from the papers "
        "as your primary source, but you may also draw on your general knowledge to elaborate, "
        "connect ideas, or fill gaps. Clearly distinguish when you are going beyond the paper "
        "by saying 'Based on general knowledge...' or 'In addition to what the paper says...' "
        "or similar markers. Always cite the source filename when referencing the papers directly. "
        "Use the conversation history when the user asks a follow-up question."
    )
    conversation_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    print("Enhanced chat mode ready (papers + external knowledge). Type a question and press Enter. Type /exit or /quit to stop.")
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting enhanced chat mode.")
            return

        if not query:
            continue
        if query.lower() in {"/exit", "/quit"}:
            print("Exiting enhanced chat mode.")
            return

        context_blocks, source_lines = build_context(store, query)
        user_prompt = (
            f"Question:\n{query}\n\n"
            "Context from papers:\n"
            + "\n\n".join(context_blocks)
        )
        answer = generate_answer(conversation_messages + [{"role": "user", "content": user_prompt}], groq_key, google_key)

        print("\nAssistant:")
        print(answer)
        if source_lines:
            print("\nDirect paper citations:")
            for line in source_lines:
                print(line)
        else:
            print("\n(No direct citations from papers in this response)")

        conversation_messages.append({"role": "user", "content": query})
        conversation_messages.append({"role": "assistant", "content": answer})



def list_groq_models(api_key: str | None = None) -> None:
    """List available Groq models using the Groq API key in env or passed in.

    Args:
        api_key: Optional API key; if None the function reads `GROQ_API_KEY`.
    """
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        print("Error: GROQ_API_KEY not set. Export it or pass it to the function.")
        return

    try:
        import groq as _groq

        client = _groq.Groq(api_key=key)
        resp = client.models.list()
        data = resp.model_dump() if hasattr(resp, "model_dump") else getattr(resp, "__dict__", {})
        models = data.get("data", []) if isinstance(data, dict) else []
        if not models:
            print("No models returned by Groq API.")
            return

        print(f"Found {len(models)} models:")
        for m in models:
            mid = m.get("id")
            active = m.get("active")
            owned = m.get("owned_by")
            ctx = m.get("context_window")
            max_tokens = m.get("max_completion_tokens")
            print(f"- {mid} | active={active} | owner={owned} | context={ctx} | max_tokens={max_tokens}")
    except Exception as exc:
        print(f"Failed to list Groq models: {exc}")


def main() -> None:
    """Parse CLI arguments and dispatch to the selected mode."""
    parser = build_parser()
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent

    if args.ingest:
        docs = ingest_papers(repo_root)
        build_index(repo_root, docs)
        return
    if args.groq_models:
        list_groq_models()
        return
    if args.chat:
        chat_papers(repo_root)
        return
    if args.chat_enhanced:
        chat_papers_enhanced(repo_root)
        return
    if args.query:
        query_papers(repo_root, args.query)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
