from datetime import date

import numpy as np

from imagepool.db import AssetDatabase


def insert_asset(database: AssetDatabase, name: str, sha: str, mime: str = "image/jpeg"):
    return database.insert(
        uuid=sha,
        original_name=name,
        mime_type=mime,
        size_bytes=10,
        width=10,
        height=10,
        sha256=sha,
        original_path=f"/{sha}.jpg",
        thumbnail_path=f"/{sha}.webp",
        embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        model_version="test:model",
    )


def test_database_insert_lookup_and_filters(tmp_path):
    database = AssetDatabase(tmp_path / "test.sqlite3")
    first = insert_asset(database, "Blue Ocean.jpg", "a")
    insert_asset(database, "Red Room.png", "b", "image/png")

    assert database.count() == 2
    assert database.find_by_hash("a") == first
    assert database.matching_ids(name="ocean") == {first.id}
    assert database.matching_ids(formats=["image/png"]) != {first.id}
    assets, total = database.list_assets(name="Blue", limit=48)
    assert total == 1
    assert assets[0].original_name == "Blue Ocean.jpg"
    assert database.matching_ids(start_date=date(2000, 1, 1)) == {1, 2}


def test_embedding_round_trip(tmp_path):
    database = AssetDatabase(tmp_path / "test.sqlite3")
    insert_asset(database, "one.jpg", "one")
    ids, embeddings = database.all_embeddings("test:model")
    assert ids.tolist() == [1]
    np.testing.assert_allclose(embeddings, [[1.0, 0.0]])

