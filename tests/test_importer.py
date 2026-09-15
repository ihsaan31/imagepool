import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from imagepool.config import Settings
from imagepool.db import AssetDatabase
from imagepool.importer import import_files, source_path
from imagepool.search import SearchIndex


def upload(name, color="blue", data=None):
    if data is None:
        buffer = io.BytesIO()
        Image.new("RGB", (20, 20), color).save(buffer, "PNG")
        data = buffer.getvalue()
    return SimpleNamespace(name=name, size=len(data), getvalue=lambda: data)


def test_folder_merge_duplicate_and_original_bytes(tmp_path):
    settings = Settings(tmp_path, "", "", 25, "test", "test")
    settings.prepare_storage()
    db = AssetDatabase(settings.database_path)
    embedder = SimpleNamespace(version="test:model", encode=lambda image: np.array([1., 0.], dtype=np.float32))
    index = SearchIndex(db, settings.index_path, embedder.version)
    collection = db.collection("Photos")
    assert db.collection("PHOTOS") == collection
    first = upload("Photos/nested/image.png")
    report = import_files([first, upload("Photos/broken.png", data=b"broken")], collection, settings, db, embedder, index)
    assert [row["Status"] for row in report] == ["Imported", "Failed"]
    asset = db.get(1)
    assert __import__('pathlib').Path(asset.original_path).read_bytes() == first.getvalue()
    assert db.asset_sources(1) == [("Photos", "Photos/nested/image.png")]
    report = import_files([first], collection, settings, db, embedder, index)
    assert report[0]["Status"] == "Duplicate"
    assert len(db.asset_sources(1)) == 1
    second_collection = db.collection("Other")
    import_files([upload("Other/image.png")], second_collection, settings, db, embedder, index)
    assert db.count() == 1
    assert db.matching_ids(collection_ids=[second_collection]) == {1}
    import_files([upload("Photos/nested/image.png", "red")], collection, settings, db, embedder, index)
    assert db.count() == 2
    assert index.index.ntotal == 2
    assert not db.matching_ids(uncollected=True)


@pytest.mark.parametrize("name", ["../bad.png", "/absolute.png", "folder/../../bad.png", "bad\x00.png"])
def test_unsafe_source_paths(name):
    with pytest.raises(ValueError):
        source_path(name)


def test_legacy_assets_survive_collection_migration(tmp_path):
    from tests.test_db import insert_asset
    db = AssetDatabase(tmp_path / "db.sqlite3")
    asset = insert_asset(db, "old.jpg", "old")
    with db.connect() as connection:
        connection.execute("DROP TABLE collection_assets")
        connection.execute("DROP TABLE collections")
    migrated = AssetDatabase(db.path)
    assert migrated.get(asset.id) == asset
    assert migrated.matching_ids(uncollected=True) == {asset.id}
