"""End-to-end vision demo on the framework core.

Usage:
    python3 scripts/run_vision.py [image_path]

If no image is provided (or the default path doesn't exist), generates a
deterministic gradient image so the demo runs offline. Top-5 ImageNet labels
are printed; profiler JSONL is appended to logs/profile.jsonl.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from PIL import Image

import vision_app  # noqa: F401 — registers stages
from npu_harness_framework import Pipeline, build


def _maybe_synthetic_image(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (224, 224))
    pixels = img.load()
    for y in range(224):
        for x in range(224):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    img.save(path, format="JPEG")
    return path


def main(argv: list[str]) -> int:
    image_path = Path(argv[1]) if len(argv) > 1 else Path("samples/sample.jpg")
    image_path = _maybe_synthetic_image(image_path)

    spec = yaml.safe_load(Path("configs/vision/imagenet_topk.yaml").read_text())
    stages = [(s["name"], build(s["stage"], s["config"])) for s in spec["stages"]]

    log_path = Path("logs/profile.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline(stages, log_path=str(log_path))

    result = pipe.run(str(image_path))

    print(f"\nTop-{len(result)} labels for {image_path}:")
    for label, prob in result:
        print(f"  {prob:.4f}  {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
