# Architecture and Design

This document describes the design of the Study RAG Assistant for
evaluators reviewing the implementation. It explains what the system
does, how it is structured, the specific design decisions that were
made, and the trade-offs behind them. Claims are anchored to source
files so the reviewer can verify each one directly.

---

## 1. Executive summary

The Study RAG Assistant is a conversational document-grounded
question-answering system. A user types a question in natural language;
the system retrieves the most semantically similar chunks from an
indexed corpus, feeds those chunks plus conversation history to a Large
Language Model under strict prompt-level constraints, and streams a
grounded answer back to the user.

Three properties separate this implementation from a minimal RAG
demonstration:

1. **Grounding is enforced at the prompt layer**, not assumed. A
   declarative YAML config and a structured prompt builder produce a
   system prompt that explicitly forbids hallucination, mandates
   verbatim quoting, and specifies a refusal protocol when context is
   insufficient. (See [§7. Anti-hallucination guarantees](#7-anti-hallucination-guarantees).)
2. **The system is conversational, not single-shot**. A summary-buffer
   memory subsystem retains recent turns verbatim and folds older turns
   into a running summary. A separate query-rewriter chain resolves
   pronouns and follow-up references before retrieval, so phrases like
   *"tell me more about that"* find the right chunks. (See [§5.4](#54-conversational-memory)
   and [§5.5](#55-query-rewriter).)
3. **Behavior is YAML-configurable**, not hard-coded. Three prompt
   chains (answerer, summarizer, rewriter) draw their system prompts
   from a single declarative configuration format. Tone, refusal
   language, reasoning strategy, and memory policy are all editable
   without touching Python. (See [§6. Configuration model](#6-configuration-model).)

The implementation comprises roughly 400 lines of Python across five
modules and 100 lines of YAML across two config files.

---

## 2. Goals and non-goals

### Goals

- **Grounded answering.** The assistant must answer from the indexed
  corpus, refuse cleanly when the corpus does not cover the question,
  and never fabricate citations.
- **Conversational coherence.** Multi-turn interactions must work:
  follow-up questions with pronouns must resolve correctly, and the
  system must recall prior context within reasonable limits.
- **Declarative tunability.** Behavioral changes (tone, refusal
  language, reasoning style, memory policy) must be possible without
  Python edits.
- **Provider agnosticism.** The system must run against OpenAI, Groq,
  or Google Gemini with no code change beyond setting an environment
  variable.
- **Demo-ready CLI.** A usable conversational interface for
  demonstration purposes, with streaming output and introspection
  commands.

### Non-goals (deliberate, not oversights)

- **Persistent cross-session memory.** Conversation memory lives in
  process memory only. Persistence to disk is straightforward to add
  but was not in scope.
- **PDF / Word native ingestion.** The system accepts `.txt` only;
  upstream conversion is expected.
- **Multi-user / web UI.** This is a single-process CLI. A web
  frontend would consume the streaming generator (`query_stream`)
  unchanged but is out of scope.
- **Production observability.** No structured logging, tracing, or
  metrics. Debugging is done at the prompt layer by inspecting YAML
  and using the `/history` slash command.
- **Fine-tuned embedding models.** The system uses an off-the-shelf
  Sentence Transformer; domain-specific fine-tuning was not pursued.

---

## 3. System overview

### Component diagram

```
                       +--------------------------+
                       |        app.py (CLI)      |
                       |  REPL, streaming, /cmds  |
                       +-----------+--------------+
                                   |
                                   v
   +-------------------------------+-----------------------------+
   |                       RAGAssistant                         |
   |   - rewriter chain   (system + history -> standalone query)|
   |   - answerer chain   (system + history + context + q)      |
   |   - summarizer chain (system + prev summary + exchanges)   |
   |   - memory state     (summary string + recent message buf) |
   +------+--------------+-----------+--------------+-----------+
          |              |           |              |
          v              v           v              v
   +-------------+  +---------+  +---------+  +-------------+
   |  VectorDB   |  | YAML    |  | Prompt  |  | LLM client  |
   |  (ChromaDB) |  | configs |  | builder |  | (OpenAI /   |
   |             |  |         |  |         |  |  Groq /     |
   |  embedding  |  |         |  |         |  |  Gemini)    |
   |  + retrieve |  |         |  |         |  |             |
   +-------------+  +---------+  +---------+  +-------------+
```

### Module responsibilities

| Module | Lines | Responsibility |
|---|---:|---|
| `src/app.py` | ~220 | CLI shell. Banner, REPL, slash commands, streaming output, spinner, signal handling. No domain logic. |
| `src/ragassistant.py` | ~230 | Orchestrator. Owns the three LCEL chains, memory state, and the `query`/`query_stream` API. |
| `src/vectordb.py` | ~115 | ChromaDB wrapper. Document ingestion (chunking + embedding) and similarity search. |
| `src/prompt_builder.py` | ~130 | Pure-functional prompt construction from config dicts. No LLM coupling. |
| `src/utils.py` | ~35 | Two helpers: `load_documents` (filesystem -> LangChain `Document`s) and `_load_yaml`. |
| `config/prompt_config.yml` | ~85 | Three prompt blocks: `rag_prompt_cfg`, `summarizer_prompt_cfg`, `rewriter_prompt_cfg`. |
| `config/config.yml` | ~50 | App-level config: reasoning strategies, memory policy. |

---

## 4. Data flow

This section traces a single user query end-to-end. Subsequent sections
go deeper on each component.

### 4.1 First-turn flow (no history)

```
1. User input arrives at the CLI REPL.
   File: src/app.py :: repl()

2. Slash command? -> Handled locally, no LLM call.
   Otherwise -> Call assistant.query_stream(question).

3. Build history list (returns []).
   File: src/ragassistant.py :: _history_messages()

4. SKIP rewriter (no history; raw question is already standalone).

5. Retrieve top-3 chunks via cosine search.
   File: src/vectordb.py :: search()

6. Join chunks with "\n\n---\n\n" separator -> context string.

7. Invoke answering chain. Stream tokens.
   Chain: SystemMessage(from YAML) -> MessagesPlaceholder("history")
          -> HumanMessage("Research Context: ... Question: ...")
          -> LLM -> StrOutputParser

8. Print tokens as they arrive (CLI).

9. After stream completes, append (HumanMessage, AIMessage) pair
   to self.recent. Compact if buffer overflowed.

Total LLM calls: 1 (answerer).
```

### 4.2 Follow-up turn flow (with history)

```
1-2. Same as above.

3. Build history list:
   [SystemMessage("Summary of earlier conversation: ...")] +
   [HumanMessage, AIMessage, HumanMessage, AIMessage, ...]

4. INVOKE rewriter:
   Input: history + raw question ("tell me more about that")
   Output: standalone search query
           ("tell me more about quantum entanglement applications")

5. Retrieve top-3 chunks using the REWRITTEN query.

6. Same as above.

7. Invoke answering chain. Chain sees the FULL history + retrieved
   context + ORIGINAL (not rewritten) question.
   Rationale: retrieval needs the explicit query; the answerer
   benefits from the user's actual phrasing for tone matching.

8-9. Same as above. Possibly triggers summarization (one extra LLM
     call) if recent buffer overflows.

Total LLM calls: 2-3 (rewriter + answerer + possibly summarizer).
```

---

## 5. Components in depth

### 5.1 Vector store

**Implementation:** `src/vectordb.py`

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional output vectors.
- Trained on broad semantic similarity tasks.
- Fast on CPU (~50ms per chunk); GPU optional via `device=` kwarg.

**Store:** ChromaDB (PersistentClient at `./chroma_db`).
- Default index: HNSW.
- Default metric: L2 (squared Euclidean) on normalized embeddings.
- Persistence: SQLite + parquet-like blob storage on disk.

**Chunking:** `RecursiveCharacterTextSplitter`
- `chunk_size=500`, `chunk_overlap=200`.
- Hierarchical separators: paragraph, line, sentence, word, character.
- Rationale: paragraph-first preserves semantic units; overlap
  ensures concepts spanning chunk boundaries remain findable.

**Ingestion (`add_documents`, lines 60-89 of `vectordb.py`)**:
- Iterates documents; chunks each; batches all chunks per document
  into a single `encode(chunks)` call (one GPU/CPU forward pass per
  document, not per chunk).
- Generates IDs of the form `{source}_{i}` where `source` is the
  document's filename and `i` is the chunk index. Falls back to a
  synthetic ID offset by `collection.count()` if `source` is missing,
  so re-runs do not collide with previously inserted documents.
- Single `collection.add(...)` call per document, reducing
  round-trips into the DB.

**Search (`search`, lines 91-115 of `vectordb.py`)**:
- Embeds the query as a single-element batch (`encode([query])`).
- Queries ChromaDB with `include=["documents", "metadatas", "distances"]`
  (note: `ids` is always returned and cannot be in `include`).
- Unwraps the `[[...]]` nesting Chroma returns (it supports
  multi-query but we always submit one), returning flat lists.

**Design decisions**:
- *Why ChromaDB?* Embedded, zero-config, persistent. No external
  service to run for a learning/demo project.
- *Why MiniLM-L6?* Strong quality-to-cost ratio. 384 dimensions keep
  the index small; CPU-friendly. Not domain-tuned, which is fine for
  general-purpose document Q&A.
- *Why chunk size 500?* Balances retrieval granularity against
  embedding noise. Smaller chunks (e.g., 200) over-fragment ideas;
  larger chunks (e.g., 1000) dilute the embedding's specificity.

### 5.2 Document ingestion

**Implementation:** `src/utils.py :: load_documents`

Walks `./data/`, loads every `.txt` file via LangChain's `TextLoader`,
and returns a list of `Document` objects. `Document` objects retain a
`metadata["source"]` field pointing back to the file, which is used
later for chunk-ID generation and for the `/topics` CLI command.

The ingestion path is idempotent against re-runs in `app.py`:

```python
existing = assistant.vector_db.collection.count()
if existing == 0:
    assistant.add_documents(docs)
```

If the collection is non-empty, ingestion is skipped. This prevents
duplicate-ID errors on repeated runs without forcing a manual
"delete chroma_db" step.

**Limitation:** Only `.txt` is supported. Adding PDF support is a
one-function change: branch on file extension in `load_documents` and
use `PyPDFLoader` for `.pdf`.

### 5.3 Prompt system

**Implementation:** `src/prompt_builder.py`, `config/prompt_config.yml`

This is the core extensibility surface. The system never hard-codes a
prompt in Python. Instead, three YAML blocks describe the three chains'
system prompts, and a single function (`build_prompt_from_config`)
turns each block into a system-message string.

**The builder (`build_prompt_from_config`, ~90 lines):**

Pure function. Takes a config dict and an optional app-level config
(for reasoning strategies). Assembles a prompt string by concatenating
labeled sections in a fixed order:

```
1. role            -> "You are <role>."
2. instruction     -> "Your task is as follows: <instruction>"
3. context         -> "Here's some background that may help you: ..."
4. output_constraints -> "Ensure your response follows these rules: ..."
5. style_or_tone   -> "Follow these style and tone guidelines: ..."
6. output_format   -> "Structure your response as follows: ..."
7. examples        -> few-shot block
8. goal            -> "Your goal is to achieve the following outcome: ..."
9. input_data      -> the user's content (optional; for runtime use)
10. reasoning_strategy -> splice in a block from app_config
11. final stinger  -> "Now perform the task as instructed above."
```

Only `instruction` is required; every other field is optional. The
fixed ordering is important because LLMs weight earlier sections more
heavily.

**Why this design instead of just writing prompts as Python strings:**

- *Declarative*. A reviewer can see what the assistant is told to do
  by reading YAML, not by tracing string concatenation in Python.
- *Reusable*. The same builder is used for all three chains
  (answerer, summarizer, rewriter) with different configs.
- *Tunable without code change*. Modifying tone, refusal language,
  or reasoning strategy is a YAML edit.
- *Composable*. Reasoning strategies live in `config.yml` and are
  spliced into prompts by name. Adding a new strategy = adding a YAML
  entry.

**The three chains in `RAGAssistant.__init__`:**

| Chain | Purpose | YAML block | Input variables |
|---|---|---|---|
| Answerer | Generate the user-facing response | `rag_prompt_cfg` | `history`, `context`, `question` |
| Summarizer | Compress old turns into a running summary | `summarizer_prompt_cfg` | `previous_summary`, `exchanges` |
| Rewriter | Turn follow-ups into standalone queries | `rewriter_prompt_cfg` | `history`, `question` |

Each is constructed identically:

```python
SYSTEM_TEXT = build_prompt_from_config(yaml_block, app_config=app_cfg)
TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_TEXT),
    MessagesPlaceholder("history"),   # for answerer + rewriter
    ("human", "...{variables}..."),
])
CHAIN = TEMPLATE | self.llm | StrOutputParser()
```

`SystemMessage(content=...)` is used (not a templated tuple) so the
builder's output is treated as literal text. This prevents stray `{`
or `}` characters in the system prompt from being interpreted as
placeholder syntax.

### 5.4 Conversational memory

**Implementation:** `src/ragassistant.py :: _history_messages`, `_record_turn`, `_compact_history`

**Strategy:** Summary buffer (the default, configurable in
`config/config.yml`).

**State:**

```python
self.summary: str               # running natural-language summary
self.recent: list[BaseMessage]  # last N turns, verbatim
```

**Per-turn flow:**

1. **Assembly** (`_history_messages`): If `summary` is non-empty,
   prepend a `SystemMessage("Summary of earlier conversation: ...")`
   before the verbatim recent messages. Return the list.

2. **Recording** (`_record_turn`): Append `HumanMessage(question)` and
   `AIMessage(answer)` to `self.recent`. If
   `len(self.recent) > buffer_size * 2`, trigger compaction.

3. **Compaction** (`_compact_history`): Take the oldest turns (all
   except the most recent `buffer_size * 2` messages), render them as
   `User: ...\nAssistant: ...` plain text, and invoke the summarizer
   chain with the current `self.summary` plus those exchanges. The
   summarizer returns an updated summary that subsumes both.

**Why summary buffer over alternatives:**

| Strategy | Pro | Con | Why not chosen |
|---|---|---|---|
| **Pure buffer** (last N verbatim) | Trivial, no extra LLM calls | Cliff-drops information older than N | Loses too much for multi-topic study sessions |
| **Pure refine** (one evolving summary) | Constant-size memory | Loses recent verbatim phrasing; coreferences break | Pronoun resolution fails immediately |
| **Summary buffer** (chosen) | Verbatim recent + summary of older | One LLM call per compaction | Best balance for the use case |

**Configurability (`config/config.yml`):**

```yaml
memory:
  strategy: summary_buffer   # or "buffer" or "none"
  buffer_size: 4
```

`buffer_size: 0` with `summary_buffer` collapses to refine mode (every
turn folded immediately). `strategy: none` disables history entirely
(stateless mode for benchmarking).

**Compaction is event-driven, not turn-counted.** Summarization fires
only when the buffer overflows, amortizing the extra LLM call across
roughly `buffer_size` turns. With the default `buffer_size: 4`, a
20-turn conversation triggers compaction ~4 times, not 20.

### 5.5 Query rewriter

**Implementation:** `src/ragassistant.py :: query_stream` (lines around the rewriter call)

**Problem it solves:**

Vector retrieval works on semantic similarity to the query string.
Conversational follow-ups like *"tell me more about that"* contain no
semantic signal — the embedding has no idea what *"that"* refers to.
Without intervention, retrieval would return arbitrary chunks.

**Mechanism:**

A separate LCEL chain (constructed from `rewriter_prompt_cfg`) sees the
conversation history plus the new question and returns a standalone
search query. The retrieval call uses the rewritten query; the
answering call uses the original question. This separation matters:

- **Retrieval needs explicit context** — *"quantum entanglement
  applications"* finds the right chunks.
- **The answerer benefits from the user's phrasing** — *"tell me more
  about that"* lets the model match conversational tone in its
  response.

**Optimization:** The rewriter is skipped on the first turn (when
`_history_messages` returns empty). Cost saved: one LLM call per fresh
session.

**Failure modes mitigated:**

- *Topic switches*: The YAML prompt explicitly tells the rewriter to
  pass through unchanged if the new message is unrelated to prior
  context. Without this, a sudden topic change would produce a
  hybrid query that retrieves wrong chunks.
- *Empty rewriter output*: A defensive `if not search_query:` clause
  falls back to the user's literal question.

### 5.6 LLM provider abstraction

**Implementation:** `src/ragassistant.py :: _initialize_llm`

Three providers supported: OpenAI, Groq, Google Gemini. The init
function checks environment variables in priority order
(`OPENAI_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY`) and returns the
first matching LangChain `ChatModel` instance.

Each provider is wrapped by an officially maintained LangChain
integration package (`langchain-openai`, `langchain-groq`,
`langchain-google-genai`). The wrappers expose the same
`Runnable` interface, so the LCEL chains (`prompt | llm | parser`) are
provider-agnostic.

Model selection per provider is overridable via env var
(`OPENAI_MODEL`, `GROQ_MODEL`, `GOOGLE_MODEL`) with sensible defaults
chosen for the cost/quality balance appropriate to a learning project.

---

## 6. Configuration model

The system has two configuration layers, separated by lifetime:

| Layer | File | Lifetime | Contents |
|---|---|---|---|
| Secrets | `.env` | Per-deployment | API keys, model overrides |
| Behavior | `config/*.yml` | Per-version-controlled | Prompts, reasoning strategies, memory policy |

**`config/prompt_config.yml`** contains three top-level blocks:

```
rag_prompt_cfg         # answerer system prompt
summarizer_prompt_cfg  # summarizer system prompt
rewriter_prompt_cfg    # rewriter system prompt
```

Each block is a config dict consumable by `build_prompt_from_config`.

**`config/config.yml`** contains:

```
reasoning_strategies   # named reasoning scaffolds (CoT, ReAct, Self-Ask, Grounded)
memory                 # memory policy: strategy + buffer_size
```

**Hot-reload:** Configs are read once at `RAGAssistant.__init__`.
Restart required for config changes. This is intentional for the
demo context — guaranteeing config invariance during a session
simplifies reasoning about behavior.

---

## 7. Anti-hallucination guarantees

Hallucination is the primary failure mode of any RAG system. This
implementation defends against it at multiple layers:

### Layer 1: Prompt-level constraints

`rag_prompt_cfg.output_constraints` in `prompt_config.yml` includes
explicit rules:

- Answer ONLY from the provided research context.
- Never invent citations, author names, statistics, dates,
  organizations, or URLs.
- When stating a fact, quote a verbatim snippet under 25 words.
- If the context does not cover the question, give a friendly refusal
  that names the topic and offers a related one. Do not fabricate.

### Layer 2: Reasoning strategy

The default reasoning strategy is `Grounded` (defined in
`config/config.yml`). It instructs the model to:

1. Identify which chunk(s) contain relevant information.
2. Extract supporting phrases verbatim. If no chunk is relevant, refuse.
3. Write the answer anchored to those phrases.
4. Audit the answer and remove any sentence not directly supported by
   an extracted phrase.

Step 4 is the critical self-check: it catches model drift where the
first half of an answer is grounded but the second half extrapolates
beyond the context.

### Layer 3: Input classification

The answerer prompt explicitly classifies the user input into three
categories before responding:

- Social pleasantry — respond warmly, do not invoke context.
- Meta-question about the conversation — answer from history.
- Research question — answer from context using the rules above.

This prevents the model from synthesizing irrelevant retrieved chunks
into an "answer" for non-research inputs like *"thanks"*.

### What this does NOT guarantee

- The model can still misread the context (interpretation errors). The
  quote requirement makes this auditable but does not eliminate it.
- The model can still refuse on questions the context does cover, if
  the retrieval surfaces the wrong chunks. (Mitigated by the rewriter
  for follow-ups; not mitigated for poorly-phrased first-turn
  questions.)
- The model can still produce a quote that does not exactly match the
  source. (Mitigated by the prompt rule "verbatim", but the model can
  paraphrase under pressure.)

---

## 8. Failure modes and mitigations

| Failure mode | Mitigation |
|---|---|
| Hallucinated citations | Prompt-level rule + Grounded reasoning strategy |
| Pronoun in follow-up retrieves wrong chunks | Query rewriter |
| Memory grows without bound | Summary buffer with event-driven compaction |
| User pastes a pleasantry / off-topic question | Three-way input classification in answerer prompt |
| Re-running app duplicates chunks | `collection.count() == 0` guard in `app.py` |
| `metadata["source"]` missing on a Document | Synthetic ID fallback in `vectordb.add_documents` |
| Empty `documents/` folder | Placeholder context string fed to LLM; prompt rules handle gracefully |
| Provider rate limit / API error during query | Caught at REPL boundary; session continues |
| User Ctrl+C mid-stream | Caught; turn is NOT recorded to memory; prompt returns |
| Rewriter returns an empty string | Defensive fallback to literal question |

---

## 9. Trade-offs and deliberate scope limits

### Cost vs. quality

The system makes 2-3 LLM calls per turn (rewriter, answerer,
occasionally summarizer). A naive single-call RAG would be cheaper but
would fail on follow-up questions and produce worse-grounded answers.
The cost increase (roughly 2-3x per turn) is judged worthwhile for the
demo use case. For production, the rewriter and summarizer could use a
smaller, cheaper model (e.g., `gpt-3.5-turbo`); this is a
single-line change.

### Memory strategy

Summary buffer was chosen over pure buffer or pure refine for the
reasons in §5.4. Tokens spent on the summary are tokens not spent on
retrieved context, so very long sessions could eventually starve
retrieval. A future enhancement would be token-budgeted compaction
(trigger on token count, not message count).

### Provider abstraction

Provider selection is environment-driven, not config-driven. This
avoids ambiguity (one source of truth: which key is set) at the cost
of being less explicit. The trade-off favors operational simplicity.

### CLI vs. web UI

A CLI was chosen for demo simplicity. The underlying `query_stream`
generator is web-frontend-ready (it streams tokens), so this decision
is reversible without architectural change.

### Token counting

Memory compaction triggers on message count, not token count. For the
default `gpt-4o-mini` (128k context), message count is sufficient. For
small-context models this could starve retrieval. The fix is to use
the already-installed `tiktoken` package to measure tokens; out of
scope for this revision.

---

## 10. Evaluation notes

This section is for reviewers grading the implementation.

### Verifiable claims

| Claim | How to verify |
|---|---|
| "Grounding is enforced at the prompt layer" | Read `config/prompt_config.yml :: rag_prompt_cfg.output_constraints` |
| "Three chains share one builder" | `src/ragassistant.py` `__init__` constructs three chains via `build_prompt_from_config` |
| "Memory compaction is event-driven" | `src/ragassistant.py :: _record_turn` triggers `_compact_history` only when `len(self.recent) > self.buffer_size * 2` |
| "Rewriter skipped on first turn" | `src/ragassistant.py :: query_stream` — `if history:` guard before rewriter invocation |
| "Provider-agnostic LCEL chains" | All chains use `... \| self.llm \| StrOutputParser()`. `self.llm` is any LangChain `ChatModel` |
| "YAML-driven, no hard-coded prompts" | `grep` for triple-quoted strings in `src/` — no instructional prose exists outside of YAML |

### How to test

```
# Grounded answer with citation
"What is quantum entanglement?"

# Pronoun resolution (rewriter exercise)
"Tell me more about that."

# Topic switch (rewriter must NOT contaminate)
"How does CRISPR work?"

# Off-topic refusal (anti-hallucination)
"How do I bake sourdough?"

# Pleasantry handling
"Thanks!"

# Meta-question (history exercise)
"What have we discussed?"

# Memory introspection (slash command)
/history
```

### What good behavior looks like

- **Grounded answers** contain at least one quoted snippet in double quotes.
- **Refusals** name the topic and offer alternatives (do not just say
  "I don't know").
- **Follow-ups** retrieve relevant chunks (verify by adding a debug
  print of `search_query` after the rewriter call).
- **Pleasantries** do NOT trigger the refusal phrase or quote
  retrieved chunks.
- **`/history`** shows verbatim turns for the last 4 exchanges and a
  summary paragraph once compaction has fired.

### What weak behavior would look like

- Answers that paraphrase the context without quoting.
- Refusals that produce the model's training-data knowledge with a
  disclaimer ("I don't have research on that, but generally...").
- Follow-up questions retrieving random chunks (rewriter
  underperforming).
- The summarizer losing entity names or numbers between compactions.
- Pleasantries returning the canned research refusal phrase.

---

## 11. File reference

### Source code

| File | Public surface |
|---|---|
| `src/app.py` | `main()` |
| `src/ragassistant.py` | `RAGAssistant`, `RAGAssistant.query`, `RAGAssistant.query_stream`, `RAGAssistant.add_documents`, `RAGAssistant.reset_memory` |
| `src/vectordb.py` | `VectorDB`, `VectorDB.add_documents`, `VectorDB.search`, `VectorDB.chunk_text` |
| `src/prompt_builder.py` | `build_prompt_from_config`, `format_prompt_section`, `print_prompt_preview` |
| `src/utils.py` | `load_documents`, `_load_yaml` |

### Configuration

| File | Top-level keys |
|---|---|
| `config/prompt_config.yml` | `rag_prompt_cfg`, `summarizer_prompt_cfg`, `rewriter_prompt_cfg` |
| `config/config.yml` | `reasoning_strategies`, `memory` |
| `.env` (not in repo) | `OPENAI_API_KEY` \| `GROQ_API_KEY` \| `GOOGLE_API_KEY`; optionally `*_MODEL` overrides |

### Runtime artifacts

| Path | Created by | Contents |
|---|---|---|
| `chroma_db/` | First run of `app.py` | ChromaDB persistent index (SQLite + binary blobs) |

---

## 12. Summary

The Study RAG Assistant is a deliberately small, deliberately
declarative implementation of a conversational document-Q&A system.
Its distinguishing features are prompt-level grounding enforcement, a
three-chain orchestration that includes a query rewriter and a
summary-buffer memory subsystem, and a YAML-driven configuration model
that makes behavior tunable without code changes.

The architecture prioritizes auditability (every prompt is in YAML,
every chain is a three-line LCEL pipeline, every memory operation is
two helper methods), provider portability (env-var-driven, three
backends supported), and conversational robustness (rewriter for
follow-ups, summary buffer for long sessions, input classification for
non-research inputs).

The trade-offs taken — 2-3 LLM calls per turn, in-memory-only state,
text-only ingestion — are aligned with the project's scope as a
learning and demonstration system. Each trade-off is documented in
§2 (non-goals) and §9 (scope limits) so reviewers can distinguish
deliberate choices from oversights.
