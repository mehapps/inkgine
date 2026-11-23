# Inkling engine runtime image: Python + the exact packages of the validated engine venv. The engine code is NOT
# baked in: compose mounts engine/ read-only at /app, so code changes need a container restart, not a rebuild.
# Build (context = this directory, ~6 GB of wheels downloaded once):
#   docker build -t inkling-engine:local -f engine/deploy/Dockerfile engine/deploy
# CPU smoke test (no GPU): docker run --rm -e INK_DEVICE=cpu -v "$PWD/engine:/app:ro" inkling-engine:local python -c "import server"
# Weights, the disk prefix cache (inkcache) and the JIT caches are mounted by compose, never copied in.
# Base pinned to the venv's interpreter (CPython 3.12.16) by digest, so a rebuild gets the same image.
FROM python:3.12.16-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

# requirements.lock = `uv pip freeze` of engine/.venv (torch 2.15.0+cu132, triton 3.9.0, xgrammar, ...).
# Regenerate it whenever the venv changes, then rebuild.
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu132 -r /tmp/requirements.lock \
 && rm /tmp/requirements.lock \
 && python -c "import torch, triton, xgrammar, fastapi, uvicorn; assert torch.__version__ == '2.15.0+cu132' and triton.__version__ == '3.9.0'"

# Triton (and inductor, for flex_attention) compile small C launchers at first use and need `cc`. Same toolchain
# as the host venv: ziglang as cc/c++ (no apt gcc, so the image matches what passed the GPU windows).
# python:3.12 images ship Python.h, which the launchers include. libcuda comes from the NVIDIA container runtime.
RUN for t in cc gcc; do printf '#!/bin/sh\nexec python -m ziglang cc "$@"\n' > /usr/local/bin/$t; done \
 && for t in c++ g++; do printf '#!/bin/sh\nexec python -m ziglang c++ "$@"\n' > /usr/local/bin/$t; done \
 && chmod +x /usr/local/bin/cc /usr/local/bin/gcc /usr/local/bin/c++ /usr/local/bin/g++

ENV CC=/usr/local/bin/cc CXX=/usr/local/bin/c++ PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
EXPOSE 30000
CMD ["python", "server.py"]
