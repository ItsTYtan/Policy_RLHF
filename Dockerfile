# 1. Base image
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

COPY requirements.txt .

RUN apt-get update
RUN apt-get install -y python3.12
RUN apt install -y python3-pip
RUN apt install python3.12-venv -y

RUN python3 -m venv venv-container \
 && ./venv-container/bin/pip install -r requirements.txt \
 && ./venv-container/bin/pip install -U packaging setuptools wheel ninja \
 && ./venv-container/bin/pip install --no-build-isolation axolotl[flash-attn,deepspeed]


# Create and switch to /app
WORKDIR /app

# Declare /app as a volume so it can be mounted at runtime
VOLUME ["/app"]

# docker build -t dev:latest .
# docker run --rm -it --gpus all -v "$(pwd)":/app dev