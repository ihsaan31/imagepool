from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class Asset:
    id: int
    uuid: str
    original_name: str
    mime_type: str
    size_bytes: int
    width: int
    height: int
    sha256: str
    original_path: str
    thumbnail_path: str
    model_version: str
    created_at: str


class AssetDatabase:
    def __init__(self, path: Path):
        self.path = path
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS assets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT NOT NULL UNIQUE,
                    original_name TEXT NOT NULL,
                    mime_type TEXT NOT NULL CHECK (mime_type IN ('image/jpeg', 'image/png')),
                    size_bytes INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    original_path TEXT NOT NULL,
                    thumbnail_path TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_assets_created_at ON assets(created_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_assets_mime_type ON assets(mime_type)"
            )
            connection.execute("CREATE TABLE IF NOT EXISTS collections (id INTEGER PRIMARY KEY, name TEXT NOT NULL, name_key TEXT NOT NULL UNIQUE)")
            connection.execute("CREATE TABLE IF NOT EXISTS collection_assets (collection_id INTEGER NOT NULL REFERENCES collections(id), asset_id INTEGER NOT NULL REFERENCES assets(id), source_path TEXT NOT NULL, PRIMARY KEY(collection_id, asset_id, source_path))")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_membership_asset ON collection_assets(asset_id)")

    def collection(self, name: str) -> int:
        name = name.strip()
        if not name or len(name) > 200:
            raise ValueError("Collection name must contain 1–200 characters.")
        with self.connect() as connection:
            connection.execute("INSERT OR IGNORE INTO collections(name, name_key) VALUES (?, ?)", (name, name.casefold()))
            return int(connection.execute("SELECT id FROM collections WHERE name_key = ?", (name.casefold(),)).fetchone()[0])

    def collections(self) -> dict[int, str]:
        with self.connect() as connection:
            return {int(row[0]): row[1] for row in connection.execute("SELECT id, name FROM collections ORDER BY name_key")}

    def link_collection(self, collection_id: int, asset_id: int, source_path: str) -> None:
        with self.connect() as connection:
            connection.execute("INSERT OR IGNORE INTO collection_assets VALUES (?, ?, ?)", (collection_id, asset_id, source_path))

    def asset_sources(self, asset_id: int) -> list[tuple[str, str]]:
        with self.connect() as connection:
            return [(row[0], row[1]) for row in connection.execute("SELECT c.name, m.source_path FROM collection_assets m JOIN collections c ON c.id = m.collection_id WHERE m.asset_id = ? ORDER BY c.name_key, m.source_path", (asset_id,))]

    @staticmethod
    def _asset(row: sqlite3.Row) -> Asset:
        return Asset(**{field: row[field] for field in Asset.__dataclass_fields__})

    def insert(
        self,
        *,
        uuid: str,
        original_name: str,
        mime_type: str,
        size_bytes: int,
        width: int,
        height: int,
        sha256: str,
        original_path: str,
        thumbnail_path: str,
        embedding: np.ndarray,
        model_version: str,
    ) -> Asset:
        vector = np.asarray(embedding, dtype=np.float32)
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO assets (
                    uuid, original_name, mime_type, size_bytes, width, height,
                    sha256, original_path, thumbnail_path, embedding,
                    embedding_dim, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid,
                    original_name,
                    mime_type,
                    size_bytes,
                    width,
                    height,
                    sha256,
                    original_path,
                    thumbnail_path,
                    vector.tobytes(),
                    vector.shape[0],
                    model_version,
                ),
            )
            row = connection.execute(
                "SELECT * FROM assets WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return self._asset(row)

    def find_by_hash(self, sha256: str) -> Asset | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM assets WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return self._asset(row) if row else None

    def get(self, asset_id: int) -> Asset | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM assets WHERE id = ?", (asset_id,)
            ).fetchone()
        return self._asset(row) if row else None

    def count(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM assets").fetchone()[0])

    def all_embeddings(self, model_version: str) -> tuple[np.ndarray, np.ndarray]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id, embedding, embedding_dim FROM assets WHERE model_version = ? ORDER BY id",
                (model_version,),
            ).fetchall()
        if not rows:
            return np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        dimensions = {row["embedding_dim"] for row in rows}
        if len(dimensions) != 1:
            raise RuntimeError("Stored embeddings have inconsistent dimensions")
        ids = np.asarray([row["id"] for row in rows], dtype=np.int64)
        vectors = np.vstack(
            [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]
        )
        return ids, vectors

    def matching_ids(
        self,
        *,
        name: str = "",
        formats: Sequence[str] = (),
        start_date: date | None = None,
        end_date: date | None = None,
        collection_ids: Sequence[int] = (),
        uncollected: bool = False,
    ) -> set[int]:
        clauses: list[str] = []
        params: list[object] = []
        if collection_ids or uncollected:
            membership = []
            if collection_ids:
                membership.append(f"id IN (SELECT asset_id FROM collection_assets WHERE collection_id IN ({','.join('?' for _ in collection_ids)}))")
                params.extend(collection_ids)
            if uncollected:
                membership.append("NOT EXISTS (SELECT 1 FROM collection_assets WHERE asset_id = assets.id)")
            clauses.append(f"({' OR '.join(membership)})")
        if name.strip():
            clauses.append("LOWER(original_name) LIKE ? ESCAPE '\\'")
            escaped = name.strip().lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.append(f"%{escaped}%")
        if formats:
            placeholders = ",".join("?" for _ in formats)
            clauses.append(f"mime_type IN ({placeholders})")
            params.extend(formats)
        if start_date:
            clauses.append("DATE(created_at) >= DATE(?)")
            params.append(start_date.isoformat())
        if end_date:
            clauses.append("DATE(created_at) <= DATE(?)")
            params.append(end_date.isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as connection:
            rows = connection.execute(f"SELECT id FROM assets {where}", params).fetchall()
        return {int(row["id"]) for row in rows}

    def list_assets(
        self,
        *,
        name: str = "",
        formats: Sequence[str] = (),
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 48,
        offset: int = 0,
        collection_ids: Sequence[int] = (),
        uncollected: bool = False,
    ) -> tuple[list[Asset], int]:
        ids = self.matching_ids(
            name=name, formats=formats, start_date=start_date, end_date=end_date,
            collection_ids=collection_ids, uncollected=uncollected,
        )
        if not ids:
            return [], 0
        placeholders = ",".join("?" for _ in ids)
        ordered_ids = sorted(ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM assets WHERE id IN ({placeholders}) ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                [*ordered_ids, limit, offset],
            ).fetchall()
        return [self._asset(row) for row in rows], len(ids)
