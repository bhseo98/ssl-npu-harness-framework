# syntax=docker/dockerfile:1.7
# CUDA 12.8 (faster-whisper/CTranslate2가 CUDA 12.x runtime을 요구). 호스트 드라이버는 13.x 호환.
ARG CUDA_TAG=12.8.0-cudnn-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# 시스템 의존성:
#   - python3.11 + venv
#   - ffmpeg (whisper용 audio 디코딩)
#   - libsndfile (soundfile)
#   - portaudio (sounddevice 마이크 캡쳐)
#   - git (model_zoo가 git pull 하는 경우 대비)

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    python3.12-dev \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    alsa-utils \
    git \
    ca-certificates \
    curl \
 && ln -sf /usr/bin/python3 /usr/local/bin/python \
 && ln -sf /usr/bin/python3 /usr/local/bin/python3 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 의존성 먼저 캐시
COPY pyproject.toml ./
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 소스
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests

# Piper voice 모델 (있으면 같이 image 에 굽고, 없으면 런타임에 마운트하거나
# RECIPES.md §15 의 curl 명령으로 받으면 됨).
COPY models ./models

# editable 재설치 (소스 복사 후)
RUN pip install -e ".[dev]"

ENV TARGET_APP_CONFIG=/workspace/configs/default.yaml

EXPOSE 8000

CMD ["python", "-m", "scripts.run_demo"]
