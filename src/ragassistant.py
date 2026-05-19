"""
Conversational RAG assistant.

The primary API is :meth:`RAGAssistant.query_with_state` — a stateless
function that takes conversation state in, returns updated state out.
This is what HTTP / NestJS callers use.

For the CLI demo, :meth:`RAGAssistant.query` and
:meth:`RAGAssistant.query_stream` are thin wrappers that hold a per-process
in-memory state, so the existing REPL keeps working unchanged.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from prompt_builder import build_prompt_from_config
from utils import _load_yaml
from vectordb import DEFAULT_CORPUS, VectorDB


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


# ----------------------------------------------------------------------
# Public data types
# ----------------------------------------------------------------------
@dataclass
class ConversationState:
    """
    Persistable conversation memory. Designed to round-trip cleanly
    through JSON so a NestJS caller can store it in a relational DB.
    """
    summary: str = ""
    # Each entry: {"role": "human" | "ai", "content": "..."}
    recent: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"summary": self.summary, "recent": list(self.recent)}

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ConversationState":
        if not data:
            return cls()
        return cls(
            summary=data.get("summary", "") or "",
            recent=list(data.get("recent", []) or []),
        )


@dataclass
class QueryResult:
    """Structured return value from :meth:`RAGAssistant.query_with_state`."""
    answer: str
    new_state: ConversationState
    sources: List[dict]
    metadata: dict


# ----------------------------------------------------------------------
# RAGAssistant
# ----------------------------------------------------------------------
class RAGAssistant:
    """
    Conversational RAG assistant. Stateless across calls when accessed
    via :meth:`query_with_state`; stateful for the CLI via :meth:`query`
    and :meth:`query_stream`.
    """

    def __init__(self):
        self.llm = self._initialize_llm()
        self.vector_db = VectorDB()

        # Load YAML configs once at startup.
        prompt_cfg_all = _load_yaml(CONFIG_DIR / "prompt_config.yml")
        app_cfg = _load_yaml(CONFIG_DIR / "config.yml")

        rag_cfg = prompt_cfg_all["rag_prompt_cfg"]
        summarizer_cfg = prompt_cfg_all["summarizer_prompt_cfg"]
        rewriter_cfg = prompt_cfg_all["rewriter_prompt_cfg"]

        memory_cfg = app_cfg.get("memory", {}) or {}
        self.memory_strategy: str = memory_cfg.get("strategy", "summary_buffer")
        self.buffer_size: int = int(memory_cfg.get("buffer_size", 4))

        # --- Main answering chain ---------------------------------------
        system_text = build_prompt_from_config(rag_cfg, app_config=app_cfg)
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_text),
            MessagesPlaceholder("history"),
            ("human",
             "Research Context:\n{context}\n\nQuestion: {question}"),
        ])
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        # --- Summarizer chain -------------------------------------------
        summarizer_text = build_prompt_from_config(
            summarizer_cfg, app_config=app_cfg
        )
        self.summarizer_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=summarizer_text),
            ("human",
             "Previous summary (may be empty):\n{previous_summary}\n\n"
             "New exchanges to fold in:\n{exchanges}\n\n"
             "Return the updated summary."),
        ])
        self.summarizer = self.summarizer_template | self.llm | StrOutputParser()

        # --- Query rewriter chain ---------------------------------------
        rewriter_text = build_prompt_from_config(
            rewriter_cfg, app_config=app_cfg
        )
        self.rewriter_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=rewriter_text),
            MessagesPlaceholder("history"),
            ("human",
             "Latest user message:\n{question}\n\n"
             "Rewrite this as a standalone search query."),
        ])
        self.rewriter = self.rewriter_template | self.llm | StrOutputParser()

        # --- CLI-mode in-memory state (NOT used by query_with_state) ----
        self._cli_state = ConversationState()
        self._cli_corpus = DEFAULT_CORPUS

        # Provider name for /health endpoint.
        self.provider_name = self.llm.__class__.__name__

        print(
            f"RAG Assistant initialized "
            f"(memory={self.memory_strategy}, buffer_size={self.buffer_size}, "
            f"provider={self.provider_name})"
        )

    # ==================================================================
    # PRIMARY (stateless) API - this is what HTTP / NestJS callers use
    # ==================================================================

    def query_with_state(
        self,
        question: str,
        state: ConversationState,
        corpus_id: str = DEFAULT_CORPUS,
        n_results: int = 3,
    ) -> QueryResult:
        """
        Stateless RAG query. Takes conversation state in, returns a
        :class:`QueryResult` with the answer and an updated state the
        caller is responsible for persisting.

        This method does NOT mutate ``self`` — multiple users can share
        a single RAGAssistant instance safely.
        """
        history_messages = self._state_to_messages(state)

        # 1. Rewrite (skip if no history).
        if history_messages:
            search_query = self.rewriter.invoke({
                "history": history_messages,
                "question": question,
            }).strip()
            if not search_query:
                search_query = question
        else:
            search_query = question

        # 2. Retrieve.
        results = self.vector_db.search(
            query=search_query,
            corpus_id=corpus_id,
            n_results=n_results,
        )
        context_chunks = results.get("documents", []) or []
        if context_chunks:
            context = "\n\n---\n\n".join(context_chunks)
        else:
            context = "(No research context was retrieved for this message.)"

        # 3. Answer.
        answer = self.chain.invoke({
            "history": history_messages,
            "context": context,
            "question": question,
        })

        # 4. Update state (immutably) + maybe compact.
        new_state = ConversationState(
            summary=state.summary,
            recent=list(state.recent),
        )
        new_state.recent.append({"role": "human", "content": question})
        new_state.recent.append({"role": "ai", "content": answer})

        compacted = False
        max_messages = self.buffer_size * 2
        if self.memory_strategy == "summary_buffer" and len(new_state.recent) > max_messages:
            new_state = self._compact_state(new_state)
            compacted = True
        elif self.memory_strategy == "buffer" and len(new_state.recent) > max_messages:
            overflow = len(new_state.recent) - max_messages
            new_state.recent = new_state.recent[overflow:]
        elif self.memory_strategy == "none":
            new_state = ConversationState()

        # 5. Build sources payload.
        sources = self._build_sources(results)

        metadata = {
            "rewritten_query": search_query if history_messages else None,
            "n_chunks_retrieved": len(context_chunks),
            "compacted": compacted,
            "provider": self.provider_name,
        }

        return QueryResult(
            answer=answer,
            new_state=new_state,
            sources=sources,
            metadata=metadata,
        )

    # ==================================================================
    # CLI-compat API - wraps the stateless API with in-memory state
    # ==================================================================

    def query(self, question: str, n_results: int = 3) -> str:
        """CLI-only: full answer as a string, using in-memory state."""
        result = self.query_with_state(
            question=question,
            state=self._cli_state,
            corpus_id=self._cli_corpus,
            n_results=n_results,
        )
        self._cli_state = result.new_state
        return result.answer

    def query_stream(self, question: str, n_results: int = 3) -> Iterator[str]:
        """
        CLI-only streaming version. Yields tokens as they arrive.
        Updates in-memory state when the stream completes.

        For HTTP callers, prefer :meth:`query_with_state` (non-streaming).
        Streaming over HTTP would require SSE and is out of scope here.
        """
        history_messages = self._state_to_messages(self._cli_state)

        if history_messages:
            search_query = self.rewriter.invoke({
                "history": history_messages,
                "question": question,
            }).strip()
            if not search_query:
                search_query = question
        else:
            search_query = question

        results = self.vector_db.search(
            query=search_query,
            corpus_id=self._cli_corpus,
            n_results=n_results,
        )
        context_chunks = results.get("documents", []) or []
        if context_chunks:
            context = "\n\n---\n\n".join(context_chunks)
        else:
            context = "(No research context was retrieved for this message.)"

        pieces: List[str] = []
        for token in self.chain.stream({
            "history": history_messages,
            "context": context,
            "question": question,
        }):
            pieces.append(token)
            yield token

        answer = "".join(pieces)

        # Update CLI state (mirrors the logic in query_with_state).
        new_state = ConversationState(
            summary=self._cli_state.summary,
            recent=list(self._cli_state.recent),
        )
        new_state.recent.append({"role": "human", "content": question})
        new_state.recent.append({"role": "ai", "content": answer})

        max_messages = self.buffer_size * 2
        if self.memory_strategy == "summary_buffer" and len(new_state.recent) > max_messages:
            new_state = self._compact_state(new_state)
        elif self.memory_strategy == "buffer" and len(new_state.recent) > max_messages:
            overflow = len(new_state.recent) - max_messages
            new_state.recent = new_state.recent[overflow:]
        elif self.memory_strategy == "none":
            new_state = ConversationState()

        self._cli_state = new_state

    def add_documents(self, documents, corpus_id: str = DEFAULT_CORPUS) -> dict:
        """Convenience wrapper around :meth:`VectorDB.add_documents`."""
        return self.vector_db.add_documents(documents=documents, corpus_id=corpus_id)

    def reset_memory(self) -> None:
        """Clear CLI in-memory state. Does not affect query_with_state callers."""
        self._cli_state = ConversationState()

    # Exposed for the CLI's /history command.
    @property
    def summary(self) -> str:
        return self._cli_state.summary

    @property
    def recent(self) -> List[BaseMessage]:
        return self._state_to_messages(self._cli_state, include_summary_message=False)

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _state_to_messages(
        self,
        state: ConversationState,
        include_summary_message: bool = True,
    ) -> List[BaseMessage]:
        """Convert a ConversationState into a list of LangChain messages."""
        if self.memory_strategy == "none":
            return []

        history: List[BaseMessage] = []
        if include_summary_message and state.summary:
            history.append(SystemMessage(
                content=f"Summary of earlier conversation:\n{state.summary}"
            ))
        for entry in state.recent:
            role = entry.get("role")
            content = entry.get("content", "")
            if role == "human":
                history.append(HumanMessage(content=content))
            elif role == "ai":
                history.append(AIMessage(content=content))
        return history

    def _compact_state(self, state: ConversationState) -> ConversationState:
        """
        Fold the oldest turns of ``state.recent`` into ``state.summary``
        via the summarizer chain. Returns a new ConversationState.
        """
        keep = self.buffer_size * 2
        to_fold = state.recent[:-keep] if keep > 0 else state.recent
        keep_recent = state.recent[-keep:] if keep > 0 else []

        exchanges = "\n".join(
            f"{'User' if e.get('role') == 'human' else 'Assistant'}: {e.get('content', '')}"
            for e in to_fold
        )
        new_summary = self.summarizer.invoke({
            "previous_summary": state.summary or "(none)",
            "exchanges": exchanges,
        }).strip()

        return ConversationState(summary=new_summary, recent=keep_recent)

    @staticmethod
    def _build_sources(search_results: dict) -> List[dict]:
        """Shape search results into a list of source dicts for API output."""
        documents = search_results.get("documents", []) or []
        metadatas = search_results.get("metadatas", []) or []
        distances = search_results.get("distances", []) or []
        ids = search_results.get("ids", []) or []

        sources = []
        for i, doc in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            sources.append({
                "text": doc,
                "source": (meta or {}).get("source"),
                "distance": distances[i] if i < len(distances) else None,
                "id": ids[i] if i < len(ids) else None,
                "metadata": meta or {},
            })
        return sources

    # ==================================================================
    # LLM provider selection
    # ==================================================================

    def _initialize_llm(self):
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=model_name,
                temperature=0.0,
            )
        if os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=model_name,
                temperature=0.0,
            )
        if os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )
        raise ValueError(
            "No valid API key found. Please set one of: "
            "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
        )
