from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import faiss
import numpy as np

from imagepool.db import AssetDatabase


class SearchIndex:
    def __init__(self, database: AssetDatabase, path: Path, model_version: str):
        self.database = database
        self.path = path
        self.metadata_path = path.with_suffix(".meta.json")
        self.model_version = model_version
        self.lock = threading.RLock()
        self.index: faiss.Index | None = None
        self._load_or_rebuild()

    def _load_or_rebuild(self) -> None:
        ids, vectors = self.database.all_embeddings(self.model_version)
        if self.path.exists() and self.metadata_path.exists():
            try:
                metadata = json.loads(self.metadata_path.read_text())
                candidate = faiss.read_index(str(self.path))
                if (
                    metadata.get("model_version") == self.model_version
                    and metadata.get("asset_count") == len(ids)
                    and candidate.ntotal == len(ids)
                ):
                    self.index = candidate
                    return
            except Exception:
                pass
        self._rebuild(ids, vectors)

    def _rebuild(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        dimension = vectors.shape[1] if vectors.ndim == 2 and len(vectors) else 512
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        if len(ids):
            index.add_with_ids(np.ascontiguousarray(vectors), ids)
        self.index = index
        self._persist()

    def _persist(self) -> None:
        if self.index is None:
            return
        temporary = self.path.with_suffix(".tmp")
        metadata_temporary = self.metadata_path.with_suffix(".tmp")
        faiss.write_index(self.index, str(temporary))
        metadata_temporary.write_text(
            json.dumps(
                {
                    "model_version": self.model_version,
                    "asset_count": int(self.index.ntotal),
                }
            )
        )
        os.replace(temporary, self.path)
        os.replace(metadata_temporary, self.metadata_path)

    def add(self, asset_id: int, embedding: np.ndarray) -> None:
        vector = np.ascontiguousarray(embedding.reshape(1, -1), dtype=np.float32)
        with self.lock:
            if self.index is None or self.index.d != vector.shape[1]:
                ids, vectors = self.database.all_embeddings(self.model_version)
                self._rebuild(ids, vectors)
                return
            self.index.add_with_ids(vector, np.asarray([asset_id], dtype=np.int64))
            self._persist()

    def refresh(self) -> None:
        """Rebuild once after a batch from the authoritative SQLite embeddings."""
        with self.lock:
            ids, vectors = self.database.all_embeddings(self.model_version)
            self._rebuild(ids, vectors)

    def search(
        self,
        embedding: np.ndarray,
        *,
        limit: int = 60,
        allowed_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            total = int(self.index.ntotal)
            k = total if allowed_ids is not None else min(limit, total)
            scores, ids = self.index.search(
                np.ascontiguousarray(embedding.reshape(1, -1), dtype=np.float32), k
            )
        matches: list[tuple[int, float]] = []
        for asset_id, score in zip(ids[0], scores[0]):
            identifier = int(asset_id)
            if identifier < 0:
                continue
            if allowed_ids is not None and identifier not in allowed_ids:
                continue
            matches.append((identifier, float(score)))
            if len(matches) == limit:
                break
        return matches
