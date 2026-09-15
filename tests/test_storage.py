import io

import pytest
from PIL import Image

from imagepool.storage import InvalidImage, persist_files, prepare_image


def image_bytes(format_name: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (80, 40), "#3857D6").save(output, format_name)
    return output.getvalue()


@pytest.mark.parametrize(
    ("format_name", "mime_type"), [("JPEG", "image/jpeg"), ("PNG", "image/png")]
)
def test_prepare_and_persist_image(tmp_path, format_name, mime_type):
    data = image_bytes(format_name)
    prepared = prepare_image(data, "unsafe/path/photo.anything", 25)
    assert prepared.original_name == "photo.anything"
    assert prepared.mime_type == mime_type
    assert (prepared.width, prepared.height) == (80, 40)

    original, thumbnail = persist_files(
        prepared, data, tmp_path / "originals", tmp_path / "thumbnails"
    )
    assert original.read_bytes() == data
    assert thumbnail.exists()


def test_invalid_image_is_rejected():
    with pytest.raises(InvalidImage):
        prepare_image(b"not an image", "fake.png", 25)


def test_oversized_image_is_rejected():
    with pytest.raises(InvalidImage):
        prepare_image(b"x" * 1025, "large.png", 0)

