FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /opt/program

# Install PyTorch nightly with Blackwell (sm_120) support
RUN uv pip install --system --no-cache \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    torch torchvision torchaudio

# Install everything else from regular PyPI
RUN uv pip install --system --no-cache \
    "numpy<2" \
    timm>=0.9.12 \
    transformers==4.38.0 \
    pycocotools>=2.0.6 \
    torchmetrics>=1.2.1 \
    tensorboard>=2.15.1 \
    albumentations>=1.3.1 \
    imageio>=2.33.1 \
    matplotlib-inline>=0.1.6 \
    matplotlib \
    opencv-python-headless>=4.8.1.78 \
    scikit-image>=0.22.0 \
    scikit-learn>=1.3.2 \
    scipy>=1.11.4 \
    shapely>=2.0.4 \
    geopandas>=1.0.1 \
    fastapi>=0.68.0 \
    uvicorn>=0.15.0 \
    python-multipart>=0.0.5 \
    tqdm>=4.62.0 \
    diskcache>=5.6.0 \
    "huggingface-hub>=0.15.1,<1.0"

RUN uv pip install --system --no-cache --no-deps buildingregulariser>=0.2.2

COPY . .

ENV PYTHONPATH=/opt/program
ENV OPENBLAS_NUM_THREADS=1

ENTRYPOINT ["./start_api.sh"]
