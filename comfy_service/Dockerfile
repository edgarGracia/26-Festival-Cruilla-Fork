FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP=pip3

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        build-essential git ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

RUN pip3 install --upgrade pip && \
    pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app

# Clone ComfyUI + custom nodes via the shared script.
COPY comfy_service/install_comfy.sh /tmp/install_comfy.sh
RUN bash /tmp/install_comfy.sh /app/Comfy

RUN pip3 install huggingface_hub loguru
COPY comfy_service /app/comfy_service

EXPOSE 8188
CMD ["python", "/app/Comfy/main.py", "--listen", "0.0.0.0", "--port", "8188"]
