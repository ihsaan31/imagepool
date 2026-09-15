from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open_clip
import torch
from PIL import Image


@dataclass
class ImageEmbedder:
    model_name: str
    pretrained: str

    def __post_init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        self.model.eval()

    @property
    def version(self) -> str:
        return f"openclip:{self.model_name}:{self.pretrained}"

    def encode(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            vector = self.model.encode_image(tensor)
            vector = vector / vector.norm(dim=-1, keepdim=True)
        return vector[0].detach().cpu().numpy().astype(np.float32)

