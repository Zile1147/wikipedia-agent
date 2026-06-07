"""
WikipediaAgent LangChain Tools
================================
Makes the WikipediaAgent available as a LangChain tool set so it can be
used inside any LangChain agent (ReAct, OpenAI Functions, etc.).

Usage::

    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_community.llms import LlamaCpp

    from wikipedia_agent.tools import make_wikipedia_tools
    from wikipedia_agent.agent import WikipediaAgent

    wiki_agent = WikipediaAgent.create(...)
    tools = make_wikipedia_tools(wiki_agent)

    llm = LlamaCpp(model_path="models/qwen-0.5b-Q4_K_M.gguf", ...)
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    result = executor.invoke({"input": "Who invented Python?"})
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .agent import WikipediaAgent


def make_wikipedia_tools(wiki_agent: "WikipediaAgent") -> List:
    """Build a list of LangChain tools wrapping *wiki_agent*.

    Returns an empty list (with a warning) if langchain is not installed,
    so the rest of the code stays importable.
    """
    try:
        from langchain_core.tools import tool
    except ImportError:
        import warnings
        warnings.warn("langchain-core not installed; tools not available.")
        return []

    @tool
    def wikipedia_search(query: str) -> str:
        """Search the offline Wikipedia knowledge base.

        Use this tool when you need factual information from Wikipedia.
        Input should be a clear, concise search query.
        Returns the top matching text chunks with article titles.
        """
        hits = wiki_agent.search(query, k=5)
        if not hits:
            return "No results found for this query."
        lines = []
        for h in hits:
            title = h.metadata.get("title", "Unknown")
            lines.append(f"[{title}] {h.text[:300]}")
        return "\n\n".join(lines)

    @tool
    def wikipedia_ask(question: str) -> str:
        """Ask a question and get a synthesized answer from Wikipedia.

        Use this tool when you want a direct answer to a factual question.
        The tool retrieves relevant Wikipedia passages and generates an answer.
        Input should be a complete question.
        """
        try:
            answer = wiki_agent.ask(question)
            return answer
        except Exception as exc:
            return f"Error generating answer: {exc}"

    @tool
    def wikipedia_get_article(title: str) -> str:
        """Retrieve the full text of a specific Wikipedia article.

        Use this when you know the exact article title you need.
        Input should be the Wikipedia article title (e.g. 'Python (programming language)').
        Returns the first 2000 characters of the article.
        """
        text = wiki_agent.get_article(title)
        if text is None:
            return f"Article '{title}' not found in the local Wikipedia archive."
        return text[:2000]

    @tool
    def wikipedia_index_article(title: str) -> str:
        """Index a specific Wikipedia article for future searches.

        Use this when a search fails and you want to add a specific article
        to the knowledge base. Input is the Wikipedia article title.
        """
        try:
            n = wiki_agent.index_article(title)
            if n == 0:
                return f"Article '{title}' not found in ZIM archive."
            return f"Successfully indexed article '{title}' ({n} chunks)."
        except Exception as exc:
            return f"Failed to index article: {exc}"

    return [wikipedia_search, wikipedia_ask, wikipedia_get_article, wikipedia_index_article]
