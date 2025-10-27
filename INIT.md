# Initial Setup Instructions

This document helps you set up the GitHub repository and DockerHub integration.

## 1. Create GitHub Repository

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: DeepSeek OCR API"

# Add remote repository
git remote add origin https://github.com/AlexXYK/deepOCR.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## 2. Configure GitHub Actions Secrets

Go to GitHub repository → Settings → Secrets and variables → Actions

Add the following secrets:

- `DOCKERHUB_USERNAME`: Your DockerHub username (alexxyk)
- `DOCKERHUB_TOKEN`: Your DockerHub access token

To generate a DockerHub token:
1. Go to DockerHub → Account Settings → Security
2. Click "New Access Token"
3. Copy the token and add as secret

## 3. Verify CI/CD Pipeline

After pushing to `main` branch:
1. Go to GitHub repository → Actions tab
2. Watch the workflow run
3. Verify successful build and push to DockerHub

## 4. Deploy to Server

### Using Portainer:

1. Open Portainer web interface
2. Go to Stacks → Add Stack
3. Name: `deepseek-ocr`
4. Build method: `Repository`
5. Repository URL: `https://github.com/AlexXYK/deepOCR.git`
6. Repository reference: `main`
7. Compose path: `docker-compose.yml`
8. Click "Deploy the stack"

### Using Docker Compose:

```bash
# SSH into your server
ssh your-server

# Pull repository
git clone https://github.com/AlexXYK/deepOCR.git
cd deepOCR

# Pull latest image
docker pull alexxyk/deepocr:latest

# Start service
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f
```

## 5. Test the Service

```bash
# Health check
curl http://your-server:5010/health

# Test OCR (replace with actual image path)
curl -X POST http://your-server:5010/ocr \
  -F "file=@/path/to/image.jpg"
```

## Next Steps

- See `README.md` for API documentation
- See `DEPLOYMENT.md` for detailed deployment guide
- Use `test_api.py` to test the service locally

## Updating the Service

After making changes:
1. Push to GitHub
2. GitHub Actions will automatically build and push to DockerHub
3. In Portainer: Update the stack
4. Or manually: `docker-compose pull && docker-compose up -d`

