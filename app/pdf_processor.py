"""
PDF processing module for converting PDF pages to images.
"""
import io
from typing import List, Tuple
from PIL import Image
import fitz  # PyMuPDF


def pdf_to_images(pdf_bytes: bytes) -> List[Tuple[Image.Image, int]]:
    """
    Convert PDF bytes to a list of PIL Images.
    
    Args:
        pdf_bytes: PDF file content as bytes
        
    Returns:
        List of tuples (PIL Image, page_number)
    """
    images = []
    
    # Open PDF from bytes
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    try:
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            
            # Render page to pixmap (image)
            # Use high DPI for better OCR quality
            zoom = 2.0  # 2x zoom for 144 DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_bytes = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_bytes))
            images.append((img, page_num + 1))
            
    finally:
        pdf_doc.close()
    
    return images

