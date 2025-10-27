"""
FastAPI application for DeepSeek OCR API.
"""
import os
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from app.pdf_processor import pdf_to_images
from app.ocr_service import OCRService

# Set CUDA device (use first GPU only)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

app = FastAPI(
    title="DeepSeek OCR API",
    description="OCR service for extracting text from images and PDFs using DeepSeek-OCR",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global OCR service instance
ocr_service = OCRService()

# Lazy load model on first request
_model_loaded = False


def ensure_model_loaded():
    """Ensure the model is loaded (lazy loading on first request)."""
    global _model_loaded
    if not _model_loaded:
        ocr_service.load_model()
        _model_loaded = True


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    print("DeepSeek OCR API starting up...")
    print(f"CUDA available: {ocr_service.device == 'cuda'}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status of the service and GPU availability
    """
    gpu_available = ocr_service.device == 'cuda'
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "model_loaded": _model_loaded
    }


@app.post("/ocr")
async def process_ocr(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Process OCR on uploaded image or PDF file.
    
    Args:
        file: Uploaded file (image or PDF)
        background_tasks: Background tasks (unused but required by FastAPI)
        
    Returns:
        JSON response with extracted text in markdown format
    """
    try:
        # Read file content
        file_bytes = await file.read()
        file_extension = file.filename.split('.')[-1].lower() if file.filename else ''
        
        images = []
        num_pages = 0
        
        # Determine if it's a PDF or image
        if file_extension == 'pdf':
            # Process PDF
            images = pdf_to_images(file_bytes)
            num_pages = len(images)
        else:
            # Process as image
            image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            images = [(image, 1)]
            num_pages = 1
        
        # Ensure model is loaded
        ensure_model_loaded()
        
        # Process all pages
        results = []
        for img, page_num in images:
            text = ocr_service.process_image(img)
            results.append({
                "page": page_num,
                "text": text
            })
        
        # Combine all page texts
        combined_text = "\n\n".join([r["text"] for r in results])
        
        return {
            "status": "success",
            "pages": num_pages,
            "text": combined_text,
            "file_type": file_extension or "unknown"
        }
        
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        API information and documentation links
    """
    return {
        "name": "DeepSeek OCR API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ocr": "/ocr",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5010)

