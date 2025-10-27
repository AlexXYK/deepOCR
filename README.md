# DeepSeek OCR API

A production-ready OCR service using DeepSeek-OCR model for extracting text from images and PDFs with markdown formatting.

## Features

- 🚀 Fast inference with GPU acceleration (NVIDIA)
- 📄 PDF support with automatic page conversion
- 🖼️ Image support (JPEG, PNG, and more)
- 📝 Markdown-formatted output preserving document layout
- 🔧 Docker containerized for easy deployment
- 🤖 Automated CI/CD via GitHub Actions → DockerHub

## Architecture

- **Framework**: FastAPI with Uvicorn
- **Model**: [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- **GPU**: Single GPU allocation (configurable)
- **Port**: 5010
- **Deployment**: Docker container with Portainer stack

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 11.8+
- Docker with NVIDIA Container Toolkit
- Portainer (optional, for stack management)

### Deployment via Portainer Stack

1. Add DockerHub credentials to Portainer secrets
2. Create new stack from `docker-compose.yml`
3. Deploy and wait for health checks to pass

```bash
# Or deploy directly with Docker Compose
docker-compose up -d
```

### Verify Deployment

```bash
# Health check
curl http://localhost:5010/health

# Or open in browser
# http://localhost:5010/docs
```

## API Endpoints

### POST `/ocr`

Process an image or PDF file for OCR.

**Request:**
```bash
curl -X POST "http://localhost:5010/ocr" \
  -H "accept: application/json" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "status": "success",
  "pages": 3,
  "text": "# Document Title\n\nContent here...",
  "file_type": "pdf"
}
```

### GET `/health`

Check service health and GPU status.

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "model_loaded": true
}
```

### GET `/docs`

Interactive API documentation (Swagger UI).

## Usage Examples

### Python Client

```python
import requests

# Process an image
with open('document.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5010/ocr',
        files={'file': f}
    )

result = response.json()
print(result['text'])
```

### cURL

```bash
# Process PDF
curl -X POST "http://localhost:5010/ocr" \
  -F "file=@document.pdf"

# Process image
curl -X POST "http://localhost:5010/ocr" \
  -F "file=@screenshot.png"
```

## Configuration

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU device to use (default: 0)
- `PYTHONUNBUFFERED`: Python output buffering

### Model Settings

The service uses "Gundam mode" for optimal quality/speed balance:
- `base_size=1024`
- `image_size=640`
- `crop_mode=True`

These settings are optimized for large documents and can be modified in `app/ocr_service.py`.

## Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run service
python app/main.py

# Or with uvicorn
uvicorn app.main:app --reload --port 5010
```

### Docker Build

```bash
# Build locally
docker build -t deepseek-ocr .

# Run locally
docker run --gpus all -p 5010:5010 deepseek-ocr
```

### Testing

```bash
# Use the test script
python test_api.py
```

## Performance

- **First request**: Model loading takes ~30-60 seconds
- **Subsequent requests**: ~2-5 seconds per page
- **GPU memory**: ~8-12GB per GPU
- **Concurrent requests**: Single GPU handles sequential processing

## Troubleshooting

### GPU Not Available

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Model Download Issues

The model will download on first request (~3GB). Ensure network access to Hugging Face.

### Memory Issues

Reduce batch size or use smaller `image_size` in `ocr_service.py`.

## License

MIT

## Acknowledgments

- [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) by DeepSeek AI
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

