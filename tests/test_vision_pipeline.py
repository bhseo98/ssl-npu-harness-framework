"""Vision pipeline end-to-end tests on the framework core.

4-stage Pipeline: LoadImage → PreprocessImagenet → ResNet18Classifier → TopK,
each a `BaseStage` from the harness core. No new ABC, no framework changes —
this file is purely a plugin user.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

import vision_app  # noqa: F401 — registers stages
from npu_harness_framework import Pipeline, build


@pytest.fixture
def synthetic_image(tmp_path) -> Path:
    img = Image.new("RGB", (224, 224))
    pixels = img.load()
    for y in range(224):
        for x in range(224):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    path = tmp_path / "sample.jpg"
    img.save(path, format="JPEG")
    return path


def test_load_image_returns_pil_rgb(synthetic_image):
    loader = build("loader", {"type": "image_file"})
    img = loader(str(synthetic_image))
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_preprocess_tensor_shape(synthetic_image):
    loader = build("loader", {"type": "image_file"})
    pre = build("preprocess", {"type": "imagenet224"})
    tensor = pre(loader(str(synthetic_image)))
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)


def test_classify_probs_shape_and_sum(synthetic_image):
    loader = build("loader", {"type": "image_file"})
    pre = build("preprocess", {"type": "imagenet224"})
    cls = build("classifier", {"type": "resnet18"})
    probs = cls(pre(loader(str(synthetic_image))))
    assert probs.shape == (1, 1000)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-3)


def test_topk_returns_sorted_k_pairs(synthetic_image):
    loader = build("loader", {"type": "image_file"})
    pre = build("preprocess", {"type": "imagenet224"})
    cls = build("classifier", {"type": "resnet18"})
    topk = build("postprocess", {"type": "topk", "k": 5})
    pairs = topk(cls(pre(loader(str(synthetic_image)))))
    assert len(pairs) == 5
    probs = [p for _, p in pairs]
    assert probs == sorted(probs, reverse=True)
    assert all(isinstance(lbl, str) for lbl, _ in pairs)


def test_e2e_pipeline_with_synthetic_image(synthetic_image):
    stages = [
        ("load", build("loader", {"type": "image_file"})),
        ("preprocess", build("preprocess", {"type": "imagenet224"})),
        ("classify", build("classifier", {"type": "resnet18"})),
        ("topk", build("postprocess", {"type": "topk", "k": 5})),
    ]
    pipe = Pipeline(stages, profiler_enabled=False)
    result = pipe.run(str(synthetic_image))
    assert isinstance(result, list) and len(result) == 5


def test_profiler_records_four_lines(synthetic_image, tmp_path):
    log = tmp_path / "profile.jsonl"
    stages = [
        ("load", build("loader", {"type": "image_file"})),
        ("preprocess", build("preprocess", {"type": "imagenet224"})),
        ("classify", build("classifier", {"type": "resnet18"})),
        ("topk", build("postprocess", {"type": "topk", "k": 5})),
    ]
    pipe = Pipeline(stages, log_path=str(log))
    pipe.run(str(synthetic_image))

    lines = log.read_text().strip().splitlines()
    assert len(lines) == 4
    recorded = [json.loads(line)["stage"] for line in lines]
    assert recorded == ["load", "preprocess", "classify", "topk"]
