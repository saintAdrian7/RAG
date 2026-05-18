import os
from typing import List
from pathlib import Path
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from prompt_builder import build_prompt_from_config
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import _load_yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class RAGAssistant:
    """
    Conversational RAG assistant with summary-buffer memory.

    Memory model
    ------------
    - ``self.recent``  - the last ``buffer_size`` USER/ASSISTANT pairs, kept
      verbatim so recent coreferences ("that paper", "the second option")
      resolve faithfully.
    - ``self.summary`` - a running natural-language summary of everything
      older than the recent buffer, produced by an LLM-driven summarizer.

    When the recent buffer grows beyond ``buffer_size``, the oldest turns
    are folded into ``summary`` and dropped from ``recent``.
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

        # Memory policy (with safe defaults if absent).
        memory_cfg = app_cfg.get("memory", {}) or {}
        self.memory_strategy: str = memory_cfg.get("strategy", "summary_buffer")
        self.buffer_size: int = int(memory_cfg.get("buffer_size", 4))

        # --- Main answering chain ---------------------------------------
        # System prompt is built once from the YAML. The history slot is
        # filled at invoke time; the human message carries the runtime
        # retrieved context and the current question.
        system_text = build_prompt_from_config(rag_cfg, app_config=app_cfg)
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_text),
            MessagesPlaceholder("history"),
            ("human",
             "Research Context:\n{context}\n\nQuestion: {question}"),
        ])
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        # --- Summarizer chain -------------------------------------------
        # A separate, narrowly-scoped LLM call whose only job is to fold
        # old turns into a running summary. Same builder, different config.
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
        # Turns conversational follow-ups ("tell me more about that") into
        # standalone search queries so retrieval can find relevant chunks
        # even when the literal user message has no semantic content on
        # its own.
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

        # --- Memory state -----------------------------------------------
        self.summary: str = ""
        self.recent: List[BaseMessage] = []

        print(
            f"RAG Assistant initialized "
            f"(memory={self.memory_strategy}, buffer_size={self.buffer_size})"
        )



    def add_documents(self, documents: List) -> None:
        """Add documents to the knowledge base."""
        self.vector_db.add_documents(documents)

    def query(self, question: str, n_results: int = 3) -> str:
        """
        Answer a user question using retrieval-augmented generation,
        with conversation history threaded in. Returns the full answer
        as a string. For token-by-token streaming, use ``query_stream``.
        """
        return "".join(self.query_stream(question, n_results=n_results))

    def query_stream(self, question: str, n_results: int = 3):
        """
        Streaming generator version of :meth:`query`. Yields answer tokens
        as they arrive from the LLM. The full answer is accumulated and
        written to conversation memory after the stream ends.

        Usage::

            for token in assistant.query_stream("..."):
                print(token, end="", flush=True)
        """
        # 1. Build a standalone search query (skip if no history yet).
        history = self._history_messages()
        if history:
            search_query = self.rewriter.invoke({
                "history": history,
                "question": question,
            }).strip()
            if not search_query:
                search_query = question
        else:
            search_query = question

        # 2. Retrieve. The YAML prompt classifies the user's input and
        #    decides whether to use the chunks, so we always pass them
        #    through - even for pleasantries that retrieve junk.
        results = self.vector_db.search(search_query, n_results=n_results)
        context_chunks = results.get("documents", [])
        if context_chunks:
            context = "\n\n---\n\n".join(context_chunks)
        else:
            context = "(No research context was retrieved for this message.)"

        # 3. Stream the answer. Accumulate pieces so we can record the
        #    full turn after streaming completes.
        pieces = []
        for token in self.chain.stream({
            "history": history,
            "context": context,
            "question": question,
        }):
            pieces.append(token)
            yield token

        # 4. Record the completed turn (and compact if buffer overflowed).
        self._record_turn(question, "".join(pieces))

    def reset_memory(self) -> None:
        """Clear conversation memory. Useful between unrelated sessions."""
        self.summary = ""
        self.recent = []

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def _history_messages(self) -> List[BaseMessage]:
        """
        Assemble the list of messages that fills the prompt's ``history``
        slot at invoke time. Order: optional summary (as a system message
        so the LLM treats it as background) followed by recent verbatim
        turns.
        """
        if self.memory_strategy == "none":
            return []

        history: List[BaseMessage] = []
        if self.summary:
            history.append(SystemMessage(
                content=f"Summary of earlier conversation:\n{self.summary}"
            ))
        history.extend(self.recent)
        return history

    def _record_turn(self, question: str, answer: str) -> None:
        """Append the new exchange to recent buffer; compact if oversized."""
        if self.memory_strategy == "none":
            return

        self.recent.append(HumanMessage(content=question))
        self.recent.append(AIMessage(content=answer))

        # Each turn = 2 messages (human + ai). buffer_size counts turns.
        max_messages = self.buffer_size * 2
        if (
            self.memory_strategy == "summary_buffer"
            and len(self.recent) > max_messages
        ):
            self._compact_history()
        elif self.memory_strategy == "buffer" and len(self.recent) > max_messages:
            # Pure buffer: drop oldest turns without summarizing.
            overflow = len(self.recent) - max_messages
            self.recent = self.recent[overflow:]

    def _compact_history(self) -> None:
        """
        Fold the oldest turns into ``self.summary`` via the summarizer chain,
        keeping only the most recent ``buffer_size`` turns verbatim.
        """
        keep = self.buffer_size * 2
        # Everything except the newest `keep` messages goes to the summarizer.
        to_fold = self.recent[:-keep] if keep > 0 else self.recent
        self.recent = self.recent[-keep:] if keep > 0 else []

        # Format the to-fold messages as plain text for the summarizer.
        exchanges = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: "
            f"{m.content}"
            for m in to_fold
        )

        new_summary = self.summarizer.invoke({
            "previous_summary": self.summary or "(none)",
            "exchanges": exchanges,
        })
        self.summary = new_summary.strip()

    # ------------------------------------------------------------------
    # LLM selection
    # ------------------------------------------------------------------

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )
