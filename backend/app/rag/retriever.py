"""
Retrieval-Augmented Generation over radiology literature.

Design:
  - Corpus: a local JSONL of curated PubMed abstracts / Radiopaedia article
    summaries / RSNA education snippets (see rag/corpus/README.md - this
    ships with a handful of seed entries; expand via scripts/build_rag_corpus.py).
  - Embeddings: sentence-transformers BioBERT-family model (domain-tuned,
    much better recall on medical terminology than a generic embedder).
  - Index: FAISS flat L2 index - fine at this corpus scale (thousands of
    docs); swap to IVF/HNSW if the corpus grows past ~100k entries.

At report-generation time we embed the findings summary and retrieve the
top-k most relevant references, which get passed into the LLM prompt as
grounding context and also surfaced in the report's citations panel.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from app.core.config import settings

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False


@dataclass
class RetrievedDoc:
    title: str
    source: str  # "pubmed" | "radiopaedia" | "rsna"
    url: str
    snippet: str
    score: float


class LiteratureRetriever:
    def __init__(self, corpus_path: str | Path | None = None):
        self.corpus_path = Path(corpus_path or Path(__file__).parent / "corpus" / "seed_corpus.jsonl")
        self.docs: list[dict] = []
        self.index = None
        self.model = None
        self._load_corpus()
        if _DEPS_AVAILABLE and self.docs:
            self._build_index()

    def _load_corpus(self):
        if not self.corpus_path.exists():
            self.docs = []
            return
        with open(self.corpus_path) as f:
            self.docs = [json.loads(line) for line in f if line.strip()]

    def _build_index(self):
        self.model = SentenceTransformer(settings.RAG_EMBED_MODEL)
        texts = [f"{d['title']}. {d['snippet']}" for d in self.docs]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.array(embeddings, dtype=np.float32))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        if not _DEPS_AVAILABLE or self.index is None or not self.docs:
            return self._keyword_fallback(query, top_k)

        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores, idxs = self.index.search(np.array(q_emb, dtype=np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.docs):
                continue
            d = self.docs[idx]
            results.append(RetrievedDoc(
                title=d["title"], source=d["source"], url=d["url"],
                snippet=d["snippet"], score=float(score),
            ))
        return results

    def _keyword_fallback(self, query: str, top_k: int) -> list[RetrievedDoc]:
        """Simple overlap scoring - used if faiss/sentence-transformers aren't installed."""
        q_terms = set(query.lower().split())
        scored = []
        for d in self.docs:
            text = f"{d['title']} {d['snippet']}".lower()
            overlap = sum(1 for t in q_terms if t in text)
            if overlap:
                scored.append((overlap, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedDoc(title=d["title"], source=d["source"], url=d["url"],
                         snippet=d["snippet"], score=float(s))
            for s, d in scored[:top_k]
        ]


_retriever_singleton: LiteratureRetriever | None = None


def get_retriever() -> LiteratureRetriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = LiteratureRetriever()
    return _retriever_singleton
