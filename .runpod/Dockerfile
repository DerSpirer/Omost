# Use NVIDIA CUDA base image with PyTorch support
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copy requirements file from parent directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir runpod

# Copy application files from parent directory
COPY lib_omost ./lib_omost
COPY chat_interface.py .
COPY .runpod/handler.py .

# Create directories for model cache
RUN mkdir -p /app/hf_download

# Set environment variables
ENV HF_HOME=/app/hf_download
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Handler for Runpod Serverless
CMD ["python", "-u", "handler.py"]

