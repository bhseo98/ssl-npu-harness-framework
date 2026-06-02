# syntax=docker/dockerfile:1.7
# vision-app — image classification on the framework core.
# CPU-only. python:3.11-slim + torch CPU wheels + torchvision + Pillow.
# Concrete plugin code lives in src/vision_app/; framework core is unchanged.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

# torch + torchvision CPU wheels (no CUDA — slim image)
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1 \
 && pip install Pillow pyyaml psutil pytest

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

RUN pip install -e .

# Pre-cache ResNet18 weights so first run is offline-friendly.
RUN python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"

CMD ["pytest", "tests/", "-v"]
