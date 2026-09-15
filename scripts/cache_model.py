"""Download the configured OpenCLIP checkpoint while building the image."""

import os

import open_clip


model_name = os.environ.get("IMAGEPOOL_MODEL_NAME", "ViT-B-32")
pretrained = os.environ.get("IMAGEPOOL_MODEL_PRETRAINED", "laion2b_s34b_b79k")

open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device="cpu")
print(f"Cached {model_name}/{pretrained}")

