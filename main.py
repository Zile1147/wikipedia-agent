#!/usr/bin/env python3
"""
WikipediaAgent — main entry point
==================================
Usage examples:

  # Build the index (first run)
  python main.py index --zim data/wikipedia_en_mini.zim --max-articles 5000

  # Ask a question
  python main.py ask "Who invented the telephone?"

  # Interactive chat
  python main.py chat

  # Show stats
  python main.py stats
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Default paths ────────────────────────────────────────────────────────────
DEFAULT_ZIM          = os.getenv("WIKI_ZIM_PATH",   "data/wikipedia_en_mini.zim")
DEFAULT_EMBED_MODEL  = os.getenv("WIKI_EMBED_MODEL", "models/gemma-embedding-270m-Q4_K_M.gguf")
DEFAULT_LLM_MODEL    = os.getenv("WIKI_LLM_MODEL",  "models/qwen-0.5b-Q4_K_M.gguf")
DEFAULT_INDEX_PATH   = os.getenv("WIKI_INDEX_PATH", "data/wikipedia_index.tvim")
DEFAULT_TOP_K        = int(os.getenv("WIKI_TOP_K",  "5"))
DEFAULT_LLM_TEMPLATE = os.getenv("WIKI_LLM_TEMPLATE", "chatml")  # chatml|deepseek


def _build_agent(args):
    """Construct WikipediaAgent from parsed CLI args."""
    # Ensure turborag is importable from sibling directory
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from wikipedia_agent import WikipediaAgent

    return WikipediaAgent.create(
        zim_path=args.zim,
        embed_model=args.embed_model,
        llm_model=args.llm_model,
        index_path=args.index_path,
        top_k=args.top_k,
        llm_template=args.llm_template,
    )


# ── Sub-commands ─────────────────────────────────────────────────────────────

def cmd_index(args):
    """Build / update the TurboRag index from the ZIM file."""
    agent = _build_agent(args)
    logger.info("Starting indexing: %s (max=%s)", args.zim, args.max_articles)
    n = agent.build_index(
        max_articles=args.max_articles,
        min_words=args.min_words,
        show_progress=True,
    )
    print(f"\n✓ Indexed {n} chunks from {args.zim}")
    print(f"  Index saved to: {args.index_path}")


def cmd_ask(args):
    """Answer a single question and exit."""
    agent = _build_agent(args)
    if agent.doc_count == 0:
        print("⚠ Index is empty.  Run: python main.py index --zim <path>")
        sys.exit(1)

    question = " ".join(args.question)
    print(f"\nQ: {question}\n")
    answer, sources = agent.ask_with_sources(question, k=args.top_k)
    print(f"A: {answer}\n")

    if args.show_sources and sources:
        print("Sources:")
        seen = set()
        for s in sources:
            title = s.metadata.get("title", s.id[:20])
            if title not in seen:
                print(f"  • {title} (score={s.score:.3f})")
                seen.add(title)


def cmd_search(args):
    """Semantic search — show raw chunks without LLM generation."""
    agent = _build_agent(args)
    query = " ".join(args.query)
    hits = agent.search(query, k=args.top_k)
    if not hits:
        print("No results found.")
        return
    for i, h in enumerate(hits, 1):
        title = h.metadata.get("title", "?")
        print(f"\n[{i}] {title}  (score={h.score:.3f})")
        print(f"    {h.text[:300]}")


def cmd_chat(args):
    """Start an interactive chat session."""
    agent = _build_agent(args)
    if agent.doc_count == 0:
        print("⚠ Index is empty.  Run: python main.py index --zim <path>")
        sys.exit(1)
    agent.chat()


def cmd_stats(args):
    """Print index statistics."""
    agent = _build_agent(args)
    import json
    print(json.dumps(agent.stats(), indent=2))


# ── CLI parser ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="wikipedia_agent",
        description="Offline Wikipedia Q&A — powered by TurboRag + llama.cpp",
    )

    # Global flags
    parser.add_argument("--zim",           default=DEFAULT_ZIM,          help="Path to .zim file")
    parser.add_argument("--embed-model",   default=DEFAULT_EMBED_MODEL,   help="Gemma embedding GGUF path")
    parser.add_argument("--llm-model",     default=DEFAULT_LLM_MODEL,     help="LLM GGUF path")
    parser.add_argument("--index-path",    default=DEFAULT_INDEX_PATH,    help="TurboVec index file (.tvim)")
    parser.add_argument("--top-k",         default=DEFAULT_TOP_K, type=int, help="Number of search results")
    parser.add_argument("--llm-template",  default=DEFAULT_LLM_TEMPLATE,  help="Chat template: chatml|deepseek|llama")

    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_idx = sub.add_parser("index", help="Build TurboRag index from ZIM file")
    p_idx.add_argument("--max-articles", type=int, default=None, help="Limit articles (default: all)")
    p_idx.add_argument("--min-words",    type=int, default=50,   help="Skip stubs shorter than this")
    p_idx.set_defaults(func=cmd_index)

    # ask
    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question", nargs="+", help="Question text")
    p_ask.add_argument("--show-sources", action="store_true", default=True)
    p_ask.set_defaults(func=cmd_ask)

    # search
    p_search = sub.add_parser("search", help="Raw semantic search (no LLM)")
    p_search.add_argument("query", nargs="+", help="Search query")
    p_search.set_defaults(func=cmd_search)

    # chat
    p_chat = sub.add_parser("chat", help="Interactive chat session")
    p_chat.set_defaults(func=cmd_chat)

    # stats
    p_stats = sub.add_parser("stats", help="Show index statistics")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
