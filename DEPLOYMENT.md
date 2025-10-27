# Deployment Guide

## Prerequisites

### Server Requirements
- NVIDIA GPU with 16GB+ VRAM
- NVIDIA drivers installed
- Docker with NVIDIA Container Toolkit
- Portainer installed (optional)

### Verify GPU Access

```bash
# Check NVIDIA driver
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Installation

### 1. Install NVIDIA Container Toolkit

```bash
# For Ubuntu 22.04
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Deployment Options

### Option A: Portainer Stack (Recommended)

1. **Setup DockerHub Secret** (if not already done)
   - Portainer → Settings → Docker → Add Registry
   - Add DockerHub credentials

2. **Deploy Stack**
   - Portainer → Stacks → Add Stack
   - Name: `deepseek-ocr`
   - Build Method: Repository
   - Repository URL: `https://github.com/AlexXYK/deepOCR`
   - Repository Reference: `main`
   - Compose path: `docker-compose.yml`
   - Click "Deploy the stack"

3. **Monitor Deployment**
   - Navigate to Containers
   - Watch logs until "Model loaded successfully"
   - Check health: `http://your-server:5010/health`

### Option B: Docker Compose

```bash
# Clone repository
git clone https://github.com/AlexXYK/deepOCR.git
cd deepOCR

# Start service
docker-compose up -d

# Check logs
docker-compose logs -f deepseek-ocr

# Check health
curl http://localhost:5010/health
```

### Option C: Standalone Docker

```bash
# Pull image from DockerHub
docker pull alexxyk/deepocr:latest

# Run container
docker run --gpus '"device=0"' \
  -p 5010:5010 \
  --restart unless-stopped \
  --name deepseek-ocr \
  alexxyk/deepocr:latest

# Check logs
docker logs -f deepseek-ocr
```

## Verification

### Check Service Health

```bash
curl http://localhost:5010/health
```

Expected response:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "model_loaded": false
}
```

### Test OCR Endpoint

```bash
# Test with an image
curl -X POST http://localhost:5010/ocr \
  -F "file=@test_image.jpg"

# Test with a PDF
curl -X POST http://localhost:5010/ocr \
  -F "file=@test_document.pdf"
```

### Access API Documentation

Open in browser:
```
http://your-server-ip:5010/docs
```

## Configuration

### Change GPU Device

Edit `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ['1']  # Use second GPU
```

### Adjust Model Settings

Edit `app/ocr_service.py`:
```python
# For higher quality (slower)
base_size=1280, image_size=1280, crop_mode=False

# For faster processing (lower quality)
base_size=640, image_size=512, crop_mode=False
```

### Volume Mount for Model Cache

Edit `docker-compose.yml`:
```yaml
volumes:
  - ./cache:/root/.cache/huggingface
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs deepseek-ocr

# Common issues:
# - GPU not available: Check nvidia-smi
# - Port already in use: Change port in docker-compose.yml
# - Out of memory: Check GPU memory with nvidia-smi
```

### Model Download Fails

```bash
# Check network connectivity
docker exec deepseek-ocr ping -c 3 huggingface.co

# Manually download model (optional)
docker exec deepseek-ocr python -c "from transformers import AutoModel; AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)"
```

### GPU Memory Issues

```bash
# Check GPU usage
nvidia-smi

# Use smaller image_size in ocr_service.py
# Or use only one GPU if multiple containers running
```

### Update Service

```bash
# Pull latest image
docker pull alexxyk/deepocr:latest

# Restart container
docker-compose restart
# or
docker restart deepseek-ocr
```

## Monitoring

### Check Service Status

```bash
# Container status
docker ps | grep deepseek-ocr

# Logs
docker logs deepseek-ocr --tail 100

# Health endpoint
curl http://localhost:5010/health
```

### GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi
```

## Security Considerations

- **Expose to internet**: Use reverse proxy (Nginx/Traefik) with SSL
- **Authentication**: Add API key authentication layer
- **Rate limiting**: Implement rate limiting middleware
- **Network**: Use firewall rules to restrict access

### Example Nginx Config

```nginx
location /api/ocr {
    proxy_pass http://localhost:5010;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    
    client_max_body_size 100M;
}
```

## Production Recommendations

1. **Use HTTPS**: Set up SSL/TLS certificate
2. **Add authentication**: Implement API key or OAuth
3. **Enable logging**: Configure structured logging
4. **Monitor metrics**: Set up Prometheus/Grafana
5. **Backup configuration**: Version control your docker-compose.yml
6. **Resource limits**: Set CPU/memory limits in docker-compose

## Support

- GitHub Issues: https://github.com/AlexXYK/deepOCR
- API Docs: http://your-server:5010/docs

