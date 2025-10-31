FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and remove flash-attn and vllm (will install separately)
COPY requirements.txt .
RUN grep -v "flash-attn" requirements.txt | grep -v "vllm" | grep -v "^--" > requirements_temp.txt && \
    pip install --no-cache-dir -r requirements_temp.txt && \
    rm requirements_temp.txt

# Install vLLM nightly for DeepSeek-OCR support
RUN pip install --no-cache-dir -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# Install flash-attn with proper CUDA setup
ENV CUDA_HOME=/usr/local/cuda
RUN pip install flash-attn==2.7.3 --no-build-isolation

# Copy application code
COPY app/ ./app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Expose port
EXPOSE 5010

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5010"]

