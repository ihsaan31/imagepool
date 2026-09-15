import numpy as np

from imagepool.db import AssetDatabase
from imagepool.search import SearchIndex


def add(database, uuid, vector):
    return database.insert(
        uuid=uuid,
        original_name=f"{uuid}.jpg",
        mime_type="image/jpeg",
        size_bytes=1,
        width=1,
        height=1,
        sha256=uuid,
        original_path=f"/{uuid}.jpg",
        thumbnail_path=f"/{uuid}.webp",
        embedding=np.asarray(vector, dtype=np.float32),
        model_version="test:model",
    )


def test_similarity_ranking_and_filter(tmp_path):
    database = AssetDatabase(tmp_path / "test.sqlite3")
    closest = add(database, "closest", [1.0, 0.0])
    other = add(database, "other", [0.0, 1.0])
    index = SearchIndex(database, tmp_path / "search.index", "test:model")

    results = index.search(np.asarray([1.0, 0.0], dtype=np.float32))
    assert results[0][0] == closest.id
    filtered = index.search(
        np.asarray([1.0, 0.0], dtype=np.float32), allowed_ids={other.id}
    )
    assert [asset_id for asset_id, _ in filtered] == [other.id]


def test_index_add_persists(tmp_path):
    database = AssetDatabase(tmp_path / "test.sqlite3")
    first = add(database, "first", [1.0, 0.0])
    index_path = tmp_path / "search.index"
    index = SearchIndex(database, index_path, "test:model")
    second = add(database, "second", [0.0, 1.0])
    index.add(second.id, np.asarray([0.0, 1.0], dtype=np.float32))

    restored = SearchIndex(database, index_path, "test:model")
    assert restored.index.ntotal == 2
    assert index_path.with_suffix(".meta.json").exists()
    assert first.id != second.id
