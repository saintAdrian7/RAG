"""
HTTP service exposing the conversational RAG assistant.

This is the stateless API that channel-specific backends (a NestJS
WhatsApp bot, a web app, mobile clients, ...) call into. The service
knows about *corpora* and *queries* — it does NOT know about users,
tenants, billing, or channels. Those concepts live in the caller.

Run locally:

    uvicorn src.server:app --reload --port 8000

Swagger UI: http://localhost:8000/docs
ReDoc:      http://localhost:8000/redoc
OpenAPI:    http://localhost:8000/openapi.json
"""

import os
import sys
from pathlib import Path


_src_dir = Path(__file__).resolve().parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

from ragassistant import ConversationState, RAGAssistant
from vectordb import DEFAULT_CORPUS


load_dotenv()

ASSISTANT: Optional[RAGAssistant] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the assistant once at startup; clean up on shutdown."""
    global ASSISTANT
    ASSISTANT = RAGAssistant()
    yield
    ASSISTANT = None


# ----------------------------------------------------------------------
# FastAPI app + OpenAPI metadata
# ----------------------------------------------------------------------
app = FastAPI(
    title="Study RAG AI Service",
    description=(
        "Stateless conversational RAG over a multi-corpus document store. The caller  is responsible for storing per-user conversation state and passing it on every request. This service is intentionally ignorant of users, tenants, billing, and channels — those concerns belong to the caller. **Authentication.** If the environment variable `AI_SERVICE_API_KEY` is set, every request must include it via the `X-API-Key` header."
    ),
    version="1.0.0",
    contact={"name": "Study RAG"},
    license_info={"name": "MIT"},
    openapi_tags=[
        {"name": "Conversation", "description": "The main RAG query endpoint."},
        {"name": "Corpora", "description": "Manage document collections."},
        {"name": "Documents", "description": "Ingest and remove documents."},
        {"name": "System", "description": "Health and liveness."},
    ],
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Auth dependency
# ----------------------------------------------------------------------
def verify_api_key(
    x_api_key: Optional[str] = Header(
        None,
        alias="X-API-Key",
        description="Shared service-to-service API key. Required only if the AI_SERVICE_API_KEY environment variable is set.",
    ),
) -> None:
    expected = os.getenv("AI_SERVICE_API_KEY", "")
    if expected and x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


def get_assistant() -> RAGAssistant:
    if ASSISTANT is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assistant not initialized",
        )
    return ASSISTANT


# ======================================================================
# Pydantic models — request/response bodies (drive the Swagger UI)
# ======================================================================

# ---- Conversation ----------------------------------------------------

class Message(BaseModel):
    """A single conversation turn. Round-trips through the caller's DB."""
    role: Literal["human", "ai"] = Field(
        ..., description="Who produced this message."
    )
    content: str = Field(..., description="The text of the message.")

    model_config = ConfigDict(json_schema_extra={
        "examples": [{"role": "human", "content": "What is photosynthesis?"}]
    })


class QueryOptions(BaseModel):
    """Tunable knobs for a single query."""
    n_results: int = Field(
        3, ge=1, le=20,
        description="Number of document chunks to retrieve from the corpus.",
    )


class QueryRequest(BaseModel):
    """The body of POST /v1/query."""
    message: str = Field(
        ..., min_length=1, max_length=10_000,
        description="The user's input message.",
    )
    history: List[Message] = Field(
        default_factory=list,
        description="Recent verbatim turns the caller has stored for this "
                    "conversation. Pass the value of `updated_recent` from "
                    "the previous response.",
    )
    summary: str = Field(
        "",
        description="Running summary of older turns. Pass the value of "
                    "`updated_summary` from the previous response.",
    )
    corpus_id: str = Field(
        DEFAULT_CORPUS, min_length=1, max_length=100,
        description="Which document corpus to retrieve from. Defaults to "
                    f"'{DEFAULT_CORPUS}'.",
    )
    options: QueryOptions = Field(default_factory=QueryOptions)

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "message": "tell me more about that",
            "history": [
                {"role": "human", "content": "What is photosynthesis?"},
                {"role": "ai", "content": "Photosynthesis is the process by which..."},
            ],
            "summary": "",
            "corpus_id": "biology_class_42",
            "options": {"n_results": 3},
        }]
    })


class Source(BaseModel):
    """One retrieved chunk that contributed to the answer."""
    text: str = Field(..., description="The chunk text.")
    source: Optional[str] = Field(
        None, description="Origin identifier (filename, URL, etc.)."
    )
    distance: Optional[float] = Field(
        None, description="Vector distance (lower = more relevant)."
    )
    id: Optional[str] = Field(None, description="Internal chunk ID.")
    metadata: dict = Field(default_factory=dict)


class QueryMetadata(BaseModel):
    """Operational metadata about how the answer was produced."""
    rewritten_query: Optional[str] = Field(
        None,
        description="The standalone query the rewriter produced from the "
                    "user's input + history. Null on the first turn (rewriter "
                    "is skipped when there is no history).",
    )
    n_chunks_retrieved: int = Field(
        ..., description="Number of chunks pulled from the corpus."
    )
    compacted: bool = Field(
        ..., description="True if the summarizer fired on this turn."
    )
    provider: str = Field(..., description="LLM provider class name.")


class QueryResponse(BaseModel):
    """The body of a successful POST /v1/query."""
    answer: str = Field(..., description="The assistant's reply.")
    updated_summary: str = Field(
        ..., description="The new running summary. Store this; pass it back "
                         "as `summary` on the next request.",
    )
    updated_recent: List[Message] = Field(
        ..., description="The new verbatim message buffer. Store this; pass "
                         "it back as `history` on the next request.",
    )
    sources: List[Source] = Field(
        ..., description="Document chunks that contributed to the answer. "
                         "Useful for displaying citations to the end user.",
    )
    metadata: QueryMetadata


# ---- Corpus / document management ------------------------------------

class CorpusInfo(BaseModel):
    corpus_id: str
    chunk_count: int = Field(..., description="Number of indexed chunks.")
    sources: List[str] = Field(
        ..., description="Distinct source identifiers in this corpus."
    )


class DocumentInput(BaseModel):
    """One document to be chunked, embedded, and indexed."""
    name: str = Field(
        ..., min_length=1,
        description="Source identifier. Used as the document's metadata "
                    "key — must be unique within a corpus or you'll get "
                    "duplicate-ID errors.",
    )
    content: str = Field(..., min_length=1, description="Document text.")
    metadata: dict = Field(
        default_factory=dict,
        description="Arbitrary metadata attached to every chunk.",
    )


class IngestRequest(BaseModel):
    documents: List[DocumentInput] = Field(..., min_length=1)

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "documents": [
                {
                    "name": "biology_ch3.txt",
                    "content": "Photosynthesis is the process...",
                    "metadata": {"chapter": 3, "subject": "biology"},
                }
            ]
        }]
    })


class IngestResponse(BaseModel):
    corpus_id: str
    documents_processed: int
    chunks_added: int


class DeleteDocumentResponse(BaseModel):
    corpus_id: str
    source: str
    chunks_removed: int


# ---- System ----------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    llm_provider: str
    memory_strategy: str
    buffer_size: int


# ======================================================================
# Endpoints
# ======================================================================

# ---- System ----------------------------------------------------------

@app.get(
    "/v1/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness probe",
    description="Returns basic information about the running assistant. Use "
                "for readiness/liveness checks in container orchestration.",
)
def health(assistant: RAGAssistant = Depends(get_assistant)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_provider=assistant.provider_name,
        memory_strategy=assistant.memory_strategy,
        buffer_size=assistant.buffer_size,
    )


# ---- Conversation ----------------------------------------------------

@app.post(
    "/v1/query",
    response_model=QueryResponse,
    tags=["Conversation"],
    summary="Ask a question",
    description=(
        "Run one conversational RAG turn against a corpus.\n\n"
        "**State.** This endpoint is stateless. The caller is responsible "
        "for persisting `updated_summary` and `updated_recent` from the "
        "response, and passing them back as `summary` and `history` on the "
        "next request for the same conversation.\n\n"
        "**Cost.** A typical turn costs 1-3 LLM calls (1 if no history, "
        "+1 for the rewriter, +1 occasionally for the summarizer).\n\n"
        "**Errors.** Returns 401 if API key is required and missing/invalid."
    ),
)
def query(
    req: QueryRequest,
    assistant: RAGAssistant = Depends(get_assistant),
    _: None = Depends(verify_api_key),
) -> QueryResponse:
    state = ConversationState(
        summary=req.summary,
        recent=[m.model_dump() for m in req.history],
    )
    result = assistant.query_with_state(
        question=req.message,
        state=state,
        corpus_id=req.corpus_id,
        n_results=req.options.n_results,
    )
    return QueryResponse(
        answer=result.answer,
        updated_summary=result.new_state.summary,
        updated_recent=[Message(**m) for m in result.new_state.recent],
        sources=[Source(**s) for s in result.sources],
        metadata=QueryMetadata(**result.metadata),
    )


# ---- Corpora --------------------------------------------------------

@app.get(
    "/v1/corpora",
    response_model=List[CorpusInfo],
    tags=["Corpora"],
    summary="List all corpora",
)
def list_corpora(
    assistant: RAGAssistant = Depends(get_assistant),
    _: None = Depends(verify_api_key),
) -> List[CorpusInfo]:
    out: List[CorpusInfo] = []
    for corpus_id in assistant.vector_db.list_corpora():
        out.append(CorpusInfo(
            corpus_id=corpus_id,
            chunk_count=assistant.vector_db.count(corpus_id),
            sources=assistant.vector_db.list_sources(corpus_id),
        ))
    return out


@app.get(
    "/v1/corpora/{corpus_id}",
    response_model=CorpusInfo,
    tags=["Corpora"],
    summary="Inspect a corpus",
    responses={404: {"description": "Corpus not found"}},
)
def get_corpus(
    corpus_id: str,
    assistant: RAGAssistant = Depends(get_assistant),
    _: None = Depends(verify_api_key),
) -> CorpusInfo:
    if corpus_id not in assistant.vector_db.list_corpora():
        raise HTTPException(status_code=404, detail="Corpus not found")
    return CorpusInfo(
        corpus_id=corpus_id,
        chunk_count=assistant.vector_db.count(corpus_id),
        sources=assistant.vector_db.list_sources(corpus_id),
    )


@app.delete(
    "/v1/corpora/{corpus_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Corpora"],
    summary="Delete a corpus",
    description="Permanently removes the corpus and every chunk in it.",
)
def delete_corpus(
    corpus_id: str,
    assistant: RAGAssistant = Depends(get_assistant),
    _: None = Depends(verify_api_key),
) -> None:
    assistant.vector_db.delete_corpus(corpus_id)


# ---- Documents -------------------------------------------------------

@app.post(
    "/v1/corpora/{corpus_id}/documents",
    response_model=IngestResponse,
    tags=["Documents"],
    summary="Ingest documents into a corpus",
    description=(
        "Chunks, embeds, and indexes documents into the named corpus. "
        "If the corpus does not exist, it is created.\n\n"
        "**Idempotency.** Document IDs are derived from `name` + chunk "
        "index. Re-ingesting a document with the same `name` will produce "
        "duplicate-ID errors. Delete the existing document first if you "
        "want to replace it."
    ),
)
def ingest_documents(
    corpus_id: str,
    req: IngestRequest,
    assistant: RAGAssistant = Depends(get_assistant),
    _: None = Depends(verify_api_key),
) -> IngestResponse:
    docs = [
        Document(
            page_content=d.content,
            metadata={**d.metadata, "source": d.name},
        )
        for d in req.documents
    ]
    result = assistant.vector_db.add_documents(documents=docs, corpus_id=corpus_id)
    return IngestResponse(
        corpus_id=corpus_id,
        documents_processed=result["documents_processed"],
        chunks_added=result["chunks_added"],
    )


@app.delete(
    "/v1/corpora/{corpus_id}/documents/{source}",
    response_model=DeleteDocumentResponse,
    tags=["Documents"],
    summary="Remove a document from a corpus",
    description="Deletes every chunk whose metadata.source matches the path "
                "parameter. Returns the number of chunks removed.",
)
def delete_document(
    corpus_id: str,
    source: str,
    assistant: RAGAssistant = Depends(get_assistant),
    _: None = Depends(verify_api_key),
) -> DeleteDocumentResponse:
    removed = assistant.vector_db.delete_document(corpus_id=corpus_id, source=source)
    return DeleteDocumentResponse(
        corpus_id=corpus_id,
        source=source,
        chunks_removed=removed,
    )
