# Study RAG Assistant

A conversational document Q&A tool built for **students and teachers**.
Drop your study materials into a folder, run one command, and ask
questions in plain English. The assistant pulls answers directly from
your documents, quotes the source text so you can verify every claim,
and remembers the conversation so follow-ups feel natural.

```
You › What are the main techniques for handling class imbalance?
Assistant › The provided research describes three techniques. "Oversampling
the minority class" rebalances training data...

You › Tell me more about the second one.
Assistant › The second technique, undersampling the majority class, works
by...
```

---

## Who it's for

- **Students** revising for exams. Drop in lecture notes, textbook
  chapters, or course PDFs (converted to text) and quiz yourself. Every
  answer comes with a quoted snippet so you know which page to revisit.
- **Teachers** building a study companion. Index your curriculum
  materials and hand students a tool that answers their questions
  before office hours, grounded only in what you taught.
- **Anyone** with a folder of documents they want conversational Q&A
  over: research papers, manuals, meeting notes, reading lists.

---

## What makes it useful

- **Grounded answers, no hallucinations.** The assistant answers
  strictly from your documents. If your notes don't cover something, it
  says so politely instead of making things up.
- **Cites with quotes.** When stating a fact, it quotes a short snippet
  from the source so you can trace every claim.
- **Handles follow-ups.** Ask *"tell me more about that"* and the
  assistant figures out what *"that"* means from earlier turns. It
  rewrites your follow-up into a self-contained search query under the
  hood.
- **Remembers across turns.** Recent exchanges stay verbatim; older ones
  are folded into a running summary, so long study sessions don't lose
  context.
- **Conversational fallbacks.** Greetings get warm replies, thanks get
  acknowledged, off-topic questions get gentle redirects, and questions
  about prior turns get answered from memory.
- **Multi-provider.** Works with OpenAI, Groq, or Google Gemini —
  whichever API key you have.
- **YAML-driven prompts.** Tune the assistant's tone, refusal language,
  or reasoning style by editing config files. No Python edits needed.

---

## Quick start

### 1. Prerequisites

- Python 3.10 or newer
- An API key for at least one of: OpenAI, Groq, or Google Gemini
  - [OpenAI](https://platform.openai.com/api-keys) — default, most polished
  - [Groq](https://console.groq.com/keys) — generous free tier
  - [Google AI Studio](https://aistudio.google.com/app/apikey) — competitive pricing

### 2. Install

```bash
git clone <your-repo-url>
cd RAG
python -m venv .venv

# Activate the virtual environment:
.venv\Scripts\activate          # Windows (cmd / PowerShell)
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

### 3. Configure your API key

Create a file named `.env` in the project root:

```
OPENAI_API_KEY=sk-...
# or
GROQ_API_KEY=...
# or
GOOGLE_API_KEY=...
```

The assistant tries OpenAI, then Groq, then Google — the first key it
finds wins. You only need one.

Optional model overrides (defaults shown):

```
OPENAI_MODEL=gpt-4o-mini
GROQ_MODEL=llama-3.1-8b-instant
GOOGLE_MODEL=gemini-2.0-flash
```

### 4. Add your documents

Put any `.txt` files you want to query into the `data/` folder. The
repository ships with sample documents covering artificial intelligence,
biotechnology, climate science, quantum computing, space exploration,
and sustainable energy. Delete or replace them with your own.

```
data/
  my_lecture_notes.txt
  textbook_chapter_3.txt
  reading_list_summary.txt
```

For PDF or Word files, export to plain text first (most word processors
have a "Save as .txt" option). Native PDF support is on the roadmap.

### 5. Run it

```bash
python src/app.py
```

The first run indexes every `.txt` in `data/` (takes ~10 seconds for
the sample set). Subsequent runs reuse the existing index stored in
`chroma_db/`.

---

## Using the chatbot

Once the prompt appears, type questions in natural language. Some things
to try:

| You ask | What happens |
|---|---|
| `What are the main approaches to climate adaptation?` | Grounded answer with a quoted snippet. |
| `Tell me more about that.` | Pronoun resolved using prior turns. |
| `What about its applications?` | Continues the thread. |
| `Thanks!` | Warm acknowledgment, no research lookup. |
| `What did we just discuss?` | Recap drawn from memory. |
| `How do I bake sourdough?` | Friendly refusal mentioning what topics are actually covered. |

### Slash commands

Type any of these at the prompt instead of a question:

| Command | What it does |
|---|---|
| `/help` | Show the command list. |
| `/topics` | List every document indexed in the corpus. |
| `/history` | Show current memory: the running summary and the recent verbatim buffer. |
| `/reset` | Clear conversation memory and start a fresh session. |
| `/quit` | Exit (also: Ctrl+C, Ctrl+D). |

`/history` is particularly useful for understanding what the assistant
"remembers" mid-conversation — great for demos and debugging.

---

## How it works

Each user question flows through this pipeline:

```
   Your question
        |
        v
 1. REWRITE      Combine the question with prior turns into a
                 self-contained search query. Skipped on turn 1.
        |
        v
 2. RETRIEVE     Embed the rewritten query, search ChromaDB for
                 the top-K most similar document chunks.
        |
        v
 3. GENERATE     The LLM sees: system instructions (from YAML),
                 conversation memory, retrieved chunks, and the
                 ORIGINAL question. Produces a grounded answer.
        |
        v
 4. REMEMBER     Append the new turn to memory. If the recent
                 buffer overflows, fold the oldest turns into a
                 running summary via a second LLM call.
        |
        v
   Streamed answer
```

Each stage is its own prompt-driven chain, configured from YAML, so you
can tune any single stage without touching the others.

---

## Project layout

```
RAG/
  src/
    app.py            CLI entry point - the chatbot REPL with streaming.
    ragassistant.py   Main assistant: wires up chains + memory.
    vectordb.py       ChromaDB wrapper for chunking + embedding + retrieval.
    prompt_builder.py Turns YAML config dicts into prompt strings.
    utils.py          Document loading + YAML helpers.
  config/
    prompt_config.yml Prompts for the answerer, summarizer, and rewriter.
    config.yml        Reasoning strategies + memory policy.
  data/               Drop your .txt documents here.
  chroma_db/          Vector index (created on first run).
  requirements.txt    Python dependencies.
  .env                Your API key(s) - you create this.
```

---

## Customizing behavior

The entire point of the YAML configs is that you can change the
assistant's personality and rules without editing Python.

### Make answers shorter, longer, or differently styled

Edit `config/prompt_config.yml` &rarr; `rag_prompt_cfg`. The
`output_constraints`, `style_or_tone`, and `output_format` fields shape
every response.

Example — to make answers terse and exam-style:

```yaml
output_constraints:
  - Keep answers under 60 words.
  - Use bullet points whenever the answer covers more than one fact.
style_or_tone:
  - Telegraphic, exam-revision style.
```

### Change refusal language

Today, when the documents don't contain an answer, the assistant gives
a friendly redirect. To make it stricter (purely refusing instead of
suggesting alternatives), edit the relevant constraint under
`rag_prompt_cfg.output_constraints`.

### Switch reasoning style

In `config/prompt_config.yml`, change `reasoning_strategy: Grounded` to
one of:

- `Grounded` (default) — anchor each claim to a specific chunk before answering.
- `CoT` — chain-of-thought; show step-by-step reasoning.
- `ReAct` — thought / action / observation / reflection loop.
- `Self-Ask` — decompose into sub-questions, then synthesize.
- `None` — no reasoning scaffold.

Strategies are defined in `config/config.yml` &rarr; `reasoning_strategies`.
You can add your own.

### Tune conversation memory

In `config/config.yml`:

```yaml
memory:
  strategy: summary_buffer    # or "buffer", or "none"
  buffer_size: 4              # number of recent user/assistant pairs kept verbatim
```

- `summary_buffer` (default) — keep last N turns verbatim; fold older into a summary via the summarizer chain.
- `buffer` — keep last N turns verbatim; drop older with no summary.
- `none` — stateless; no conversation history fed to the LLM.

Set `buffer_size: 0` with `summary_buffer` for pure "refine" mode (every
turn folded into the summary immediately — useful for very long
sessions).

### Add a different document type

Document loading lives in `src/utils.py` &rarr; `load_documents`. It
currently uses LangChain's `TextLoader` for `.txt` files. To support
PDFs, install `pypdf` and add a branch that uses
`PyPDFLoader(file_path)` for files ending in `.pdf`.

---

## Cost note

A typical turn uses 2-3 LLM calls:

- One **rewriter** call (skipped on the first turn).
- One **answerer** call (this is the main one).
- One **summarizer** call, fired roughly every `buffer_size` turns.

On `gpt-4o-mini` that's fractions of a cent per question. Groq's free
tier covers casual use without billing. Gemini Flash is similar in cost
to OpenAI.

If cost matters, set `OPENAI_MODEL=gpt-4o-mini` (already the default)
and increase `buffer_size` in `config.yml` to reduce summarizer fire
frequency.

---

## Limitations and roadmap

Things this version doesn't do yet:

- **No PDF / Word native support.** Convert to `.txt` first.
- **No cross-session memory.** Memory lives in process memory; quitting
  wipes it. Saving `assistant.summary` + `assistant.recent` to JSON
  would fix this.
- **No source highlighting in the UI.** The model quotes snippets, but
  the CLI doesn't show *which file* each snippet came from. The
  metadata is in the vector DB; surfacing it is a small enhancement.
- **CPU embeddings.** Indexing uses the CPU by default. For large
  corpora (thousands of pages) pass `device="cuda"` to
  `SentenceTransformer` in `vectordb.py`.

---

## Troubleshooting

**"No valid API key found"** — Make sure `.env` exists in the project
root (not inside `src/`) and contains a key on its own line, no quotes.

**The same document keeps getting re-indexed** — `chroma_db/` may have
been deleted between runs. Delete it intentionally to force a re-index;
otherwise the assistant detects existing chunks via
`collection.count() > 0`.

**Answers feel too generic / not grounded enough** — Likely a prompt
issue. Open `config/prompt_config.yml` and tighten the
`output_constraints` under `rag_prompt_cfg`. Specifically, the rules
about quoting snippets and refusing when context is insufficient.

**Follow-up questions return unrelated chunks** — The rewriter may be
over- or under-expanding. Add `print(f"[rewrite] {search_query!r}")` in
`ragassistant.py` right after the rewriter call to see what query is
actually hitting the vector DB; adjust the `rewriter_prompt_cfg` in
YAML based on what you see.

**Out-of-context error / token limit exceeded** — Lower `n_results` in
the `query()` call or reduce `buffer_size`. Each retrieved chunk and
each remembered turn competes for the model's context window.

---

## Architecture in one paragraph

A `ChatPromptTemplate` with a `MessagesPlaceholder` for conversation
history is piped through the configured LLM via LangChain's LCEL
(`prompt | llm | StrOutputParser()`). Three such chains run in concert:
a **rewriter** that turns conversational follow-ups into standalone
search queries, an **answerer** that does the actual RAG generation, and
a **summarizer** that compresses old turns into a running summary. All
three pull their system prompts from YAML via `build_prompt_from_config`,
so the entire personality of the assistant is declarative. The vector
store is ChromaDB with `sentence-transformers/all-MiniLM-L6-v2`
embeddings, persisted to `./chroma_db`.

That's it. The whole system is about 400 lines of Python plus 100
lines of YAML.
