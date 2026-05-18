"""
Demo CLI for the conversational RAG assistant.

Run from the project root:

    python src/app.py

Inside the prompt, type natural questions about your indexed research,
or use one of the slash commands listed by /help.
"""

import itertools
import os
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from ragassistant import RAGAssistant
from utils import load_documents


# ----------------------------------------------------------------------
# ANSI styling helpers
# ----------------------------------------------------------------------
# Modern Windows terminals (Windows Terminal, PowerShell 7, VSCode,
# Git Bash) all support ANSI. On legacy cmd.exe, calling os.system("")
# flips on VT processing for the rest of the session.
if os.name == "nt":
    os.system("")

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"


def c(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


def system_msg(text: str) -> None:
    print(c(text, DIM))


def error_msg(text: str) -> None:
    print(c(text, RED))


# ----------------------------------------------------------------------
# Spinner shown while the rewriter + retrieval run (the silent gap
# before answer tokens start streaming).
# ----------------------------------------------------------------------
class Spinner:
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str = "thinking"):
        self.message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def _run() -> None:
            frames = itertools.cycle(self.FRAMES)
            while not self._stop.is_set():
                sys.stdout.write(f"\r{c(next(frames) + ' ' + self.message + '…', DIM)}")
                sys.stdout.flush()
                time.sleep(0.08)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=0.3)
        # Clear the spinner line.
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()


# ----------------------------------------------------------------------
# UI bits
# ----------------------------------------------------------------------
def banner() -> None:
    title = "Research RAG Assistant"
    width = 60
    pad = (width - len(title) - 2) // 2
    line = "─" * width
    print(c(f"┌{line}┐", CYAN, BOLD))
    print(c(f"│{' ' * pad}{title}{' ' * (width - len(title) - pad)}│", CYAN, BOLD))
    print(c(f"└{line}┘", CYAN, BOLD))
    system_msg("Ask questions about your indexed research. Type /help for commands.\n")


def print_help() -> None:
    rows = [
        ("/help", "Show this help"),
        ("/topics", "List the source files indexed in the corpus"),
        ("/history", "Show current memory (summary + recent turns)"),
        ("/reset", "Clear conversation memory"),
        ("/quit", "Exit (also: Ctrl+C, Ctrl+D)"),
    ]
    print(c("Commands:", CYAN, BOLD))
    for cmd, desc in rows:
        print(f"  {c(cmd, BOLD):<20} {c(desc, DIM)}")
    print()


def list_topics(assistant: RAGAssistant) -> None:
    """Pull unique source filenames from the collection's metadata."""
    try:
        raw = assistant.vector_db.collection.get(include=["metadatas"])
        metadatas = raw.get("metadatas", []) or []
        sources = sorted({
            Path(m["source"]).stem
            for m in metadatas
            if m and "source" in m
        })
        if not sources:
            system_msg("No documents indexed yet.")
            return
        print(c("Topics in the corpus:", CYAN, BOLD))
        for s in sources:
            print(f"  • {s}")
        print()
    except Exception as exc:
        error_msg(f"Couldn't list topics: {exc}")


def show_history(assistant: RAGAssistant) -> None:
    """Pretty-print the current memory state for demo transparency."""
    if not assistant.summary and not assistant.recent:
        system_msg("No conversation history yet.")
        return

    if assistant.summary:
        print(c("Running summary of older turns:", CYAN, BOLD))
        print(f"  {assistant.summary}\n")

    if assistant.recent:
        print(c(f"Recent verbatim buffer ({len(assistant.recent) // 2} turn(s)):", CYAN, BOLD))
        for msg in assistant.recent:
            role = "You" if isinstance(msg, HumanMessage) else "Asst"
            content = msg.content.replace("\n", " ")
            if len(content) > 120:
                content = content[:117] + "…"
            print(f"  {c('[' + role + ']', DIM)} {content}")
        print()


# ----------------------------------------------------------------------
# Streaming answer with spinner-until-first-token
# ----------------------------------------------------------------------
def stream_answer(assistant: RAGAssistant, question: str) -> None:
    """
    Run the streaming query. Keep the spinner alive until the first
    token arrives (covers the rewrite + retrieve gap), then swap to
    live token output.
    """
    spinner = Spinner("thinking")
    spinner.start()
    first = True
    try:
        for token in assistant.query_stream(question):
            if first:
                spinner.stop()
                sys.stdout.write(c("Assistant › ", GREEN, BOLD))
                sys.stdout.flush()
                first = False
            sys.stdout.write(token)
            sys.stdout.flush()
        if first:
            # The chain yielded nothing - rare, but handle it.
            spinner.stop()
            error_msg("(no response generated)")
        else:
            sys.stdout.write("\n\n")
            sys.stdout.flush()
    finally:
        # Belt and suspenders: ensure the spinner thread is stopped
        # even if streaming raised mid-flight.
        spinner.stop()


# ----------------------------------------------------------------------
# REPL
# ----------------------------------------------------------------------
SLASH_COMMANDS = {"/help", "/topics", "/history", "/reset", "/quit", "/exit"}


def repl(assistant: RAGAssistant) -> None:
    while True:
        try:
            raw = input(c("You › ", BLUE, BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print()  # newline so the goodbye sits cleanly
            break

        if not raw:
            continue

        lower = raw.lower()

        # Slash commands.
        if lower in ("/quit", "/exit", "quit", "exit"):
            break
        if lower == "/help":
            print_help()
            continue
        if lower == "/topics":
            list_topics(assistant)
            continue
        if lower == "/history":
            show_history(assistant)
            continue
        if lower == "/reset":
            assistant.reset_memory()
            system_msg("Conversation memory cleared.\n")
            continue
        if lower.startswith("/"):
            error_msg(f"Unknown command: {raw}. Type /help.\n")
            continue

        # Real question.
        try:
            stream_answer(assistant, raw)
        except KeyboardInterrupt:
            print()
            system_msg("(interrupted - your question was not recorded)\n")
            continue
        except Exception as exc:
            error_msg(f"\nError handling question: {exc}\n")
            continue

    system_msg("Goodbye!")


# ----------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------
def main() -> None:
    load_dotenv()
    banner()

    try:
        system_msg("Initializing…")
        assistant = RAGAssistant()

        existing = assistant.vector_db.collection.count()
        if existing == 0:
            system_msg("Loading documents from ./data …")
            docs = load_documents()
            assistant.add_documents(docs)
            system_msg(
                f"Indexed {assistant.vector_db.collection.count()} chunks "
                f"from {len(docs)} document(s)."
            )
        else:
            system_msg(f"Using existing collection ({existing} chunks already indexed).")
        print()

        repl(assistant)

    except Exception as exc:
        error_msg(f"Fatal error: {exc}")
        error_msg("Make sure ./config exists and your .env has an API key:")
        error_msg("  OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY")
        sys.exit(1)


if __name__ == "__main__":
    main()
