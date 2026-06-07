"""
WikipediaAgent
==============
A fully offline Wikipedia question-answering agent.

Architecture:
  ┌──────────────────────────────────────────────────────────┐
  │  User question                                           │
  │       ↓                                                  │
  │  Gemma 270M Embedder  (via llama.cpp)                   │
  │       ↓                                                  │
  │  TurboRag search  (TurboVec Q4_K_M index)               │
  │       ↓                                                  │
  │  Top-k Wikipedia chunks (from .zim archive)             │
  │       ↓                                                  │
  │  Qwen 0.5B / DeepSeek 1.3B (RAG answer via llama.cpp)  │
  │       ↓                                                  │
  │  Answer + Sources                                        │
  └──────────────────────────────────────────────────────────┘

Usage::

    agent = WikipediaAgent.create(
        zim_path="wikipedia_en_mini.zim",
        embed_model="models/gemma-embedding-270m-Q4_K_M.gguf",
        llm_model="models/qwen-0.5b-Q4_K_M.gguf",
    )

    # Build index (only needed once — saved to disk)
    agent.build_index(max_articles=5000)

    # Ask questions offline
    answer = agent.ask("Who invented the telephone?")
    print(answer)

    # Interactive loop
    agent.chat()
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    zim_path: str = "data/wikipedia_en_mini.zim"
    embed_model: str = "models/gemma-embedding-270m-Q4_K_M.gguf"
    llm_model: str = "models/qwen-0.5b-Q4_K_M.gguf"
    index_path: str = "data/wikipedia_index.tvim"
    docstore_path: str = "data/wikipedia_docs.db"
    embed_dim: int = 2048
    bit_width: int = 4
    top_k: int = 5
    chunk_size: int = 400
    chunk_overlap: int = 80
    llm_template: str = "chatml"     # "chatml" for Qwen, "deepseek" for DeepSeek


class WikipediaAgent:
    """Offline Wikipedia Q&A agent powered by TurboRag + llama.cpp.

    Parameters
    ----------
    rag : TurboRag
        Pre-built TurboRag instance (with index loaded from disk).
    config : AgentConfig
        Runtime settings.
    """

    SYSTEM_PROMPT = (
        "You are a knowledgeable Wikipedia assistant. "
        "Answer questions accurately and concisely based only on the provided context. "
        "If the context does not contain enough information, say so clearly. "
        "Do not make up facts."
    )

    def __init__(self, rag, config: AgentConfig) -> None:
        self.rag = rag
        self.config = config
        self._history: List[dict] = []

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        zim_path: str = "data/wikipedia_en_mini.zim",
        embed_model: str = "models/gemma-embedding-270m-Q4_K_M.gguf",
        llm_model: str = "models/qwen-0.5b-Q4_K_M.gguf",
        index_path: str = "data/wikipedia_index.tvim",
        embed_dim: int = 2048,
        top_k: int = 5,
        llm_template: str = "chatml",
    ) -> "WikipediaAgent":
        """Create the agent — loads models lazily on first call."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from turborag import TurboRag
        from turborag.config import (
            TurboRagConfig, EmbedderConfig, LLMConfig, IndexConfig
        )
        from turborag.llm import LLM

        cfg = TurboRagConfig(
            embedder=EmbedderConfig(model_path=embed_model, dim=embed_dim),
            llm=LLMConfig(model_path=llm_model, chat_template=llm_template),
            index=IndexConfig(
                dim=embed_dim,
                bit_width=4,
                index_path=index_path,
                docstore_path=index_path.replace(".tvim", "_docs.db"),
            ),
            top_k=top_k,
        )
        llm = LLM.from_config(cfg.llm)
        rag = TurboRag(config=cfg, llm=llm)

        agent_cfg = AgentConfig(
            zim_path=zim_path,
            embed_model=embed_model,
            llm_model=llm_model,
            index_path=index_path,
            embed_dim=embed_dim,
            top_k=top_k,
            llm_template=llm_template,
        )
        return cls(rag=rag, config=agent_cfg)

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(
        self,
        max_articles: Optional[int] = None,
        min_words: int = 50,
        show_progress: bool = True,
    ) -> int:
        """Ingest the ZIM file into the TurboRag index.

        Safe to call multiple times — checkpointing ensures already-indexed
        articles are not re-processed.

        Parameters
        ----------
        max_articles : int, optional
            Limit to this many articles (useful for testing).
        min_words : int
            Skip stubs shorter than this.
        show_progress : bool
            Print progress every 100 articles.

        Returns
        -------
        int
            Total number of text chunks indexed.
        """
        from .indexer import ZimIndexer
        if not os.path.exists(self.config.zim_path):
            raise FileNotFoundError(
                f"ZIM file not found: {self.config.zim_path}\n"
                "Download from: https://download.kiwix.org/zim/wikipedia/"
            )
        indexer = ZimIndexer(
            zim_path=self.config.zim_path,
            rag=self.rag,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        n_chunks = indexer.index_all(
            max_articles=max_articles,
            min_words=min_words,
            show_progress=show_progress,
        )
        return n_chunks

    # ------------------------------------------------------------------
    # Q&A
    # ------------------------------------------------------------------

    def ask(self, question: str, k: Optional[int] = None) -> str:
        """Answer a question using the Wikipedia index.

        Returns the answer string.
        """
        top_k = k or self.config.top_k
        answer, sources = self.rag.ask(question, k=top_k, system=self.SYSTEM_PROMPT)
        self._history.append({"question": question, "answer": answer})
        return answer

    def ask_with_sources(self, question: str, k: Optional[int] = None):
        """Answer a question and return (answer, sources) tuple."""
        top_k = k or self.config.top_k
        answer, sources = self.rag.ask(question, k=top_k, system=self.SYSTEM_PROMPT)
        self._history.append({"question": question, "answer": answer})
        return answer, sources

    def search(self, query: str, k: Optional[int] = None):
        """Semantic search without generation — returns raw hits."""
        return self.rag.search(query, k=k or self.config.top_k)

    def get_article(self, title: str) -> Optional[str]:
        """Retrieve a specific Wikipedia article's plain text."""
        from .zim_reader import ZimReader
        with ZimReader(self.config.zim_path) as r:
            article = r.get_article(title)
            return article.text if article else None

    def index_article(self, title: str) -> int:
        """Index a single article on demand (lazy indexing)."""
        from .indexer import ZimIndexer
        indexer = ZimIndexer(self.config.zim_path, self.rag)
        return indexer.index_article(title)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def doc_count(self) -> int:
        return self.rag.doc_count

    def stats(self) -> dict:
        return {
            **self.rag.stats(),
            "zim_path": self.config.zim_path,
            "history_turns": len(self._history),
        }

    # ------------------------------------------------------------------
    # Interactive chat loop
    # ------------------------------------------------------------------

    def chat(self) -> None:
        """Run an interactive command-line chat session."""
        print("\n" + "="*60)
        print("  WikipediaAgent — Offline Wikipedia Q&A")
        print(f"  Index: {self.doc_count} chunks | Model: {self.config.llm_model}")
        print("  Type 'quit' to exit, 'stats' for index info")
        print("="*60 + "\n")

        while True:
            try:
                question = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if question.lower() == "stats":
                import json
                print(json.dumps(self.stats(), indent=2))
                continue

            print("Searching …", end="\r")
            answer, sources = self.ask_with_sources(question)
            print(f"\nAgent: {answer}\n")

            if sources:
                print("Sources:")
                seen = set()
                for s in sources:
                    title = s.metadata.get("title", s.id)
                    if title not in seen:
                        print(f"  • {title}")
                        seen.add(title)
            print()
