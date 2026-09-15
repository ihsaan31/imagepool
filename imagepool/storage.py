from __future__ import annotations

import hashlib
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError


ALLOWED_FORMATS = {"JPEG": ("image/jpeg", ".jpg"), "PNG": ("image/png", ".png")}


class InvalidImage(ValueError):
    pass


@dataclass(frozen=True)
class PreparedImage:
    uuid: str
    original_name: str
    mime_type: str
    extension: str
    size_bytes: int
    width: int
    height: int
    sha256: str
    image: Image.Image


def prepare_image(data: bytes, original_name: str, max_upload_mb: int) -> PreparedImage:
    if not data:
        raise InvalidImage("The file is empty.")
    if len(data) > max_upload_mb * 1024 * 1024:
        raise InvalidImage(f"File exceeds the {max_upload_mb} MB limit.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(data)) as decoded:
            detected_format = decoded.format
            if detected_format not in ALLOWED_FORMATS:
                raise InvalidImage("Only JPEG and PNG files are accepted.")
            mime_type, extension = ALLOWED_FORMATS[detected_format]
            oriented = ImageOps.exif_transpose(decoded).convert("RGB")
            oriented.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise InvalidImage("The file is not a valid JPEG or PNG image.") from error
    return PreparedImage(
        uuid=str(uuid4()),
        original_name=Path(original_name).name or f"image{extension}",
        mime_type=mime_type,
        extension=extension,
        size_bytes=len(data),
        width=oriented.width,
        height=oriented.height,
        sha256=hashlib.sha256(data).hexdigest(),
        image=oriented,
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".upload-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def persist_files(
    prepared: PreparedImage,
    original_data: bytes,
    originals_dir: Path,
    thumbnails_dir: Path,
) -> tuple[Path, Path]:
    original_path = originals_dir / f"{prepared.uuid}{prepared.extension}"
    thumbnail_path = thumbnails_dir / f"{prepared.uuid}.webp"
    thumbnail = prepared.image.copy()
    thumbnail.thumbnail((640, 640), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    thumbnail.save(buffer, "WEBP", quality=84, method=6)
    _atomic_write(original_path, original_data)
    try:
        _atomic_write(thumbnail_path, buffer.getvalue())
    except Exception:
        original_path.unlink(missing_ok=True)
        raise
    return original_path, thumbnail_path

