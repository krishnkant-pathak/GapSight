from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
from google import genai
from google.genai.errors import APIError

from app.core.config import settings

logger = logging.getLogger(__name__)


_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[Any] = None
_seed_lock: asyncio.Lock = asyncio.Lock()


def _get_collection() -> Any:
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    _chroma_client = chromadb.PersistentClient(path=settings.vector_db_path)
    _collection = _chroma_client.get_or_create_collection(
        name=settings.vector_db_collection,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "Chroma collection '%s' ready at %s",
        settings.vector_db_collection,
        settings.vector_db_path,
    )
    return _collection


def _resolved_api_key() -> Optional[str]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip().lower() in {"", "replace-me", "your-key-here"}:
        return None
    return api_key.strip()


async def _embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    if not texts:
        return []

    api_key = _resolved_api_key()
    if api_key is None:
        logger.warning("GOOGLE_API_KEY not configured; skipping embedding call.")
        return None

    client = genai.Client(api_key=api_key)
    try:
        response = await client.aio.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=texts,
        )
    except APIError as exc:
        logger.exception("Gemini embedding call failed: %s", exc)
        return None
    except Exception as exc:
        logger.exception("Unexpected error during Gemini embedding call: %s", exc)
        return None

    if not response.embeddings:
        logger.error("Gemini embedding response had no embeddings.")
        return None

    return [list(emb.values) for emb in response.embeddings]


async def seed_vector_db(patent_corpus: List[Dict[str, Any]]) -> None:
    if not patent_corpus:
        logger.info("seed_vector_db called with empty corpus; nothing to do.")
        return

    async with _seed_lock:
        collection = await asyncio.to_thread(_get_collection)
        existing = await asyncio.to_thread(collection.count)
        if existing > 0:
            logger.info(
                "Vector DB already seeded with %d patent(s); skipping reseed.",
                existing,
            )
            return

        abstracts = [str(p.get("abstract", "")).strip() for p in patent_corpus]
        filtered = [
            (p, abstracts[i])
            for i, p in enumerate(patent_corpus)
            if abstracts[i]
        ]
        if not filtered:
            logger.error("No usable seed records; vector DB will remain empty.")
            return

        embeddings = await _embed_texts([abstract for _, abstract in filtered])
        if embeddings is None:
            logger.error(
                "Could not generate embeddings (missing key or API error); "
                "vector DB seeding aborted."
            )
            return

        ids = [str(p["patent_id"]) for p, _ in filtered]
        documents = [abstract for _, abstract in filtered]
        metadatas = [
            {
                "patent_id": str(p["patent_id"]),
                "title": str(p.get("title", "")),
            }
            for p, _ in filtered
        ]

        await asyncio.to_thread(
            collection.upsert,
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Seeded vector DB with %d patent(s).", len(ids))


async def search_prior_art(query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if not query_text or not query_text.strip():
        return []

    embeddings = await _embed_texts([query_text.strip()])
    if not embeddings:
        return []

    try:
        collection = await asyncio.to_thread(_get_collection)
        result = await asyncio.to_thread(
            collection.query,
            query_embeddings=embeddings,
            n_results=top_k,
        )
    except Exception as exc:
        logger.exception("Vector DB query failed: %s", exc)
        return []

    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]

    hits: List[Dict[str, Any]] = []
    for idx, doc_id in enumerate(ids):
        distance = float(dists[idx]) if idx < len(dists) else 1.0
        meta = metas[idx] if idx < len(metas) else {}
        document = docs[idx] if idx < len(docs) else ""
        hits.append(
            {
                "patent_id": str(meta.get("patent_id") or doc_id),
                "title": str(meta.get("title", "")),
                "abstract": str(document),
                "similarity_score": round(max(0.0, 1.0 - distance), 4),
            }
        )
    return hits
