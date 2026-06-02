"""Four vision stages — each is a `BaseStage` plugged into the harness core.

Pipeline shape:  path/str  →  PIL.Image  →  torch.Tensor (1,3,224,224)
                 →  softmax probs (1,1000)  →  list[(label, prob)]
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18

from npu_harness_framework import BaseStage, register


@register("loader", "image_file")
class LoadImage(BaseStage):
    def __call__(self, path: str | Path) -> Image.Image:
        return Image.open(path).convert("RGB")


@register("preprocess", "imagenet224")
class PreprocessImagenet(BaseStage):
    def __init__(self) -> None:
        self._tf = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self._tf(img).unsqueeze(0)


@register("classifier", "resnet18")
class ResNet18Classifier(BaseStage):
    def __init__(self) -> None:
        self._weights = ResNet18_Weights.IMAGENET1K_V1
        self._model = resnet18(weights=self._weights).eval()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._model(x).softmax(dim=-1)


@register("postprocess", "topk")
class TopK(BaseStage):
    def __init__(self, k: int = 5) -> None:
        self._labels = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
        self._k = k

    def __call__(self, probs: torch.Tensor) -> list[tuple[str, float]]:
        top = probs[0].topk(self._k)
        return [(self._labels[int(i)], float(v)) for v, i in zip(top.values, top.indices)]
