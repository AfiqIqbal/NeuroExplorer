from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from ..database.models import FileAnalysis as DBAnalysis
from ..database.models import db

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="NeuroExplorer UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class FileItem(BaseModel):
    id: int
    file_name: str
    file_path: str
    file_type: str
    summary: Optional[str]
    tags: List[str]
    agent: Optional[str]
    extra_metadata: dict
    score: Optional[float] = None


class MetaResponse(BaseModel):
    agents: List[str]
    file_types: List[str]
    total: int


def get_session() -> Iterable[Session]:
    session = db.get_session()
    try:
        yield session
    finally:
        session.close()



class EmbeddingProvider:
    """Generates normalized query embeddings for supported models."""

    TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self) -> None:
        self._text_model: SentenceTransformer | None = None
        self._clip_model: CLIPModel | None = None
        self._clip_processor: CLIPProcessor | None = None
        self._cached_dims: Dict[str, int] = {}

    def _ensure_text(self) -> SentenceTransformer:
        if self._text_model is None:
            self._text_model = SentenceTransformer(self.TEXT_MODEL_NAME)
            sample = self._text_model.encode(["sample"], convert_to_numpy=True)[0]
            self._cached_dims["text"] = int(sample.shape[0])
        return self._text_model

    def get_text_dim(self) -> int:
        self._ensure_text()
        return self._cached_dims["text"]

    def _ensure_clip(self) -> tuple[CLIPModel, CLIPProcessor]:
        if self._clip_model is None or self._clip_processor is None:
            self._clip_processor = CLIPProcessor.from_pretrained(self.CLIP_MODEL_NAME)
            self._clip_model = CLIPModel.from_pretrained(self.CLIP_MODEL_NAME)
            self._clip_model.eval()
            self._cached_dims["clip"] = int(self._clip_model.config.projection_dim)
        return self._clip_model, self._clip_processor

    def get_clip_dim(self) -> int:
        self._ensure_clip()
        return self._cached_dims["clip"]

    def text_embedding(self, text: str) -> Optional[np.ndarray]:
        model = self._ensure_text()
        vec = model.encode([text])[0]
        return normalize_embedding(vec)

    def clip_text_embedding(self, text: str) -> Optional[np.ndarray]:
        model, processor = self._ensure_clip()
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            features = model.get_text_features(**inputs)
        vec = features.squeeze(0).cpu().numpy()
        return normalize_embedding(vec)

    def get_query_embedding(self, text: str, embedding_type: str, expected_dim: int) -> Optional[np.ndarray]:
        if embedding_type == "clip":
            vec = self.clip_text_embedding(text)
        else:
            vec = self.text_embedding(text)

        if vec is None:
            return None
        if expected_dim and vec.shape[0] != expected_dim:
            return None
        return vec


embedding_provider = EmbeddingProvider()


IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    "node_modules",
}


SIMILARITY_THRESHOLDS: Dict[str, float] = {
    "clip": 0.20,
    "text": 0.30,
}


def resolve_agent(record: DBAnalysis) -> Optional[str]:
    for metadata in (record.extra_metadata, record.metadata):
        if isinstance(metadata, dict) and metadata.get("agent"):
            return str(metadata["agent"])
    return None


def infer_embedding_type(agent_name: Optional[str], dimension: int) -> str:
    if agent_name and "image" in agent_name.lower():
        return "clip"
    if agent_name and "text" in agent_name.lower():
        return "text"

    # Fall back on dimension-based heuristics
    try:
        clip_dim = embedding_provider.get_clip_dim()
    except Exception:
        clip_dim = None
    if clip_dim and dimension == clip_dim:
        return "clip"
    try:
        text_dim = embedding_provider.get_text_dim()
    except Exception:
        text_dim = None
    if text_dim and dimension == text_dim:
        return "text"
    return "text"


def normalize_path_key(path: str) -> str:
    return Path(path).as_posix().lower()


def normalize_embedding(embedding: Optional[List[float]]) -> Optional[np.ndarray]:
    if embedding is None:
        return None
    vec = np.asarray(embedding, dtype=np.float32)
    if vec.size == 0:
        return None
    norm = np.linalg.norm(vec)
    if norm == 0 or np.isnan(norm):
        return None
    return vec / norm


def record_to_item(record: DBAnalysis, score: Optional[float] = None) -> FileItem:
    return FileItem(
        id=record.id,
        file_name=record.file_name,
        file_path=record.file_path,
        file_type=record.file_type,
        summary=record.summary,
        tags=record.tags or [],
        agent=resolve_agent(record),
        extra_metadata=record.extra_metadata or {},
        score=score,
    )


@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found. Please build the static assets.")
    return FileResponse(index_path)


@app.get("/api/files", response_model=List[FileItem])
def list_files(
    agent: Optional[str] = Query(None, description="Filter by agent name"),
    file_type: Optional[str] = Query(None, description="Filter by file extension"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    include_ignored: bool = Query(False, description="Include files from ignored directories such as .venv/"),
    session: Session = Depends(get_session),
) -> List[FileItem]:
    query = session.query(DBAnalysis).order_by(DBAnalysis.updated_at.desc())
    records = query.offset(offset).limit(limit * 3).all()  # fetch extra for post-filtering

    filtered: List[FileItem] = []
    seen_paths: set[str] = set()
    for record in records:
        if not include_ignored:
            parts = set(Path(record.file_path).parts)
            if not parts.isdisjoint(IGNORED_DIRECTORIES):
                continue
        if agent and resolve_agent(record) != agent:
            continue
        if file_type and record.file_type != file_type:
            continue
        key = normalize_path_key(record.file_path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        filtered.append(record_to_item(record))
        if len(filtered) >= limit:
            break

    return filtered


@app.get("/api/meta", response_model=MetaResponse)
def get_meta(session: Session = Depends(get_session)) -> MetaResponse:
    records = session.query(DBAnalysis).all()
    agents = sorted({resolve_agent(r) for r in records if resolve_agent(r)})
    file_types = sorted({r.file_type for r in records if r.file_type})
    return MetaResponse(agents=agents, file_types=file_types, total=len(records))


@app.get("/api/search", response_model=List[FileItem])
def search_files(
    query: str = Query(..., min_length=1, description="Text to search for"),
    agent: Optional[str] = Query(None),
    file_type: Optional[str] = Query(None),
    top_k: int = Query(10, ge=1, le=50),
    include_ignored: bool = Query(False),
    session: Session = Depends(get_session),
) -> List[FileItem]:
    try:
        records = session.query(DBAnalysis).all()

        candidates = []
        fallback_candidates = []
        query_cache: Dict[tuple[str, int], Optional[np.ndarray]] = {}
        seen_paths: set[str] = set()
        for record in records:
            if not include_ignored:
                parts = set(Path(record.file_path).parts)
                if not parts.isdisjoint(IGNORED_DIRECTORIES):
                    continue
            if agent and resolve_agent(record) != agent:
                continue
            if file_type and record.file_type != file_type:
                continue
            doc_vec = normalize_embedding(record.embedding)
            if doc_vec is None:
                continue
            key = normalize_path_key(record.file_path)
            if key in seen_paths:
                continue
            dimension = int(doc_vec.shape[0])
            embedding_type = infer_embedding_type(resolve_agent(record), dimension)
            cache_key = (embedding_type, dimension)
            if cache_key not in query_cache:
                query_cache[cache_key] = embedding_provider.get_query_embedding(query, embedding_type, dimension)
            query_vec = query_cache[cache_key]
            if query_vec is None or doc_vec.shape != query_vec.shape:
                continue
            seen_paths.add(key)
            score = float(np.dot(query_vec, doc_vec))
            threshold = SIMILARITY_THRESHOLDS.get(embedding_type, 0.0)
            entry = (record, score)
            if score >= threshold:
                candidates.append(entry)
            else:
                fallback_candidates.append(entry)

        pool = candidates if candidates else fallback_candidates
        pool.sort(key=lambda pair: pair[1], reverse=True)
        top_candidates = pool[:top_k]
        return [record_to_item(record, score=score) for record, score in top_candidates]

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc


@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok"}
