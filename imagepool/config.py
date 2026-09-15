from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    username: str
    password_hash: str
    max_upload_mb: int
    model_name: str
    model_pretrained: str

    @property
    def database_path(self) -> Path:
        return self.data_dir / "imagepool.sqlite3"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "search.index"

    @property
    def originals_dir(self) -> Path:
        return self.data_dir / "assets" / "originals"

    @property
    def thumbnails_dir(self) -> Path:
        return self.data_dir / "assets" / "thumbnails"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.environ.get("IMAGEPOOL_DATA_DIR", "./data")).resolve(),
            username=os.environ.get("IMAGEPOOL_USERNAME", ""),
            password_hash=os.environ.get("IMAGEPOOL_PASSWORD_HASH", ""),
            max_upload_mb=int(os.environ.get("IMAGEPOOL_MAX_UPLOAD_MB", "25")),
            model_name=os.environ.get("IMAGEPOOL_MODEL_NAME", "ViT-B-32"),
            model_pretrained=os.environ.get(
                "IMAGEPOOL_MODEL_PRETRAINED", "laion2b_s34b_b79k"
            ),
        )

    def prepare_storage(self) -> None:
        for directory in (self.data_dir, self.originals_dir, self.thumbnails_dir):
            directory.mkdir(parents=True, exist_ok=True)
        probe = self.data_dir / ".write-probe"
        probe.write_bytes(b"ok")
        probe.unlink()

