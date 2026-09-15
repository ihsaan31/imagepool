"""Per-file durable folder import; originals are never re-encoded."""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Callable

from imagepool.storage import prepare_image, persist_files


def source_path(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or "\x00" in normalized or not path.name:
        raise ValueError("Invalid source path.")
    return str(path)


def import_files(files, collection_id, settings, database, embedder, search_index,
                 on_progress: Callable | None = None):
    if len(files) > 200 or sum(file.size for file in files) > 512 * 1024 * 1024:
        raise ValueError("Import limit: 200 images and 512 MB total.")
    results = []
    try:
        for position, file in enumerate(files):
            original = thumbnail = asset = None
            try:
                relative_path = source_path(file.name)
                data = file.getvalue()
                prepared = prepare_image(data, file.name, settings.max_upload_mb)
                asset = database.find_by_hash(prepared.sha256)
                status = "Duplicate"
                if asset is None:
                    vector = embedder.encode(prepared.image)
                    original, thumbnail = persist_files(prepared, data, settings.originals_dir, settings.thumbnails_dir)
                    asset = database.insert(
                        uuid=prepared.uuid, original_name=prepared.original_name,
                        mime_type=prepared.mime_type, size_bytes=prepared.size_bytes,
                        width=prepared.width, height=prepared.height, sha256=prepared.sha256,
                        original_path=str(original), thumbnail_path=str(thumbnail),
                        embedding=vector, model_version=embedder.version,
                    )
                    status = "Imported"
                if collection_id is not None:
                    database.link_collection(collection_id, asset.id, relative_path)
                results.append({"File": file.name, "Status": status, "Detail": ""})
            except Exception as error:
                if asset is None:
                    if original:
                        original.unlink(missing_ok=True)
                    if thumbnail:
                        thumbnail.unlink(missing_ok=True)
                results.append({"File": file.name, "Status": "Failed", "Detail": str(error)})
            if on_progress:
                on_progress((position + 1) / len(files), file.name)
    finally:
        # Also runs on interrupted processing; failed persistence is repaired at restart.
        search_index.refresh()
    return results
