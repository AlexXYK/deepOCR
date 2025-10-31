"""
OCR service for DeepSeek-OCR model using transformers.
"""
import torch
import re
from PIL import Image
from typing import List, Optional
from enum import Enum
from transformers import AutoProcessor, AutoModelForCausalLM

try:
    from bs4 import BeautifulSoup
    BEAUTIFUL_SOUP_AVAILABLE = True
except ImportError:
    BEAUTIFUL_SOUP_AVAILABLE = False


class OCRMode(str, Enum):
    """OCR mode configurations."""
    TINY = "tiny"
    SMALL = "small"
    BASE = "base"
    LARGE = "large"
    GUNDAM = "gundam"


class PromptType(str, Enum):
    """Predefined prompt types for different OCR tasks."""
    DOCUMENT = "document"
    FREE_OCR = "free_ocr"
    FIGURES = "figures"
    DESCRIBE = "describe"
    OTHER = "other"


PROMPT_TEMPLATES = {
    PromptType.DOCUMENT: "<image>\n<|grounding|>Convert the document to markdown.",
    PromptType.FREE_OCR: "<image>\nFree OCR.",
    PromptType.FIGURES: "<image>\nParse the figure.",
    PromptType.DESCRIBE: "<image>\nDescribe this image in detail.",
    PromptType.OTHER: "<image>\nOCR this image."
}


def html_table_to_markdown(html: str) -> str:
    """Convert HTML table to markdown table."""
    if not BEAUTIFUL_SOUP_AVAILABLE:
        return html
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        if not table:
            return html
        
        rows = []
        for tr in table.find_all('tr'):
            cells = []
            for td in tr.find_all(['td', 'th']):
                cell_text = td.get_text(strip=True)
                cells.append(cell_text)
            if cells:
                rows.append(cells)
        
        if not rows:
            return html
        
        max_cols = max(len(row) for row in rows)
        
        for row in rows:
            while len(row) < max_cols:
                row.append('')
        
        markdown_lines = []
        markdown_lines.append('| ' + ' | '.join(rows[0]) + ' |')
        markdown_lines.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
        
        for row in rows[1:]:
            markdown_lines.append('| ' + ' | '.join(row) + ' |')
        
        return '\n'.join(markdown_lines)
    except Exception as e:
        print(f"Error converting table: {e}")
        return html


def clean_ocr_output(text: str) -> str:
    """Clean OCR output by removing metadata tags."""
    if not text:
        return ""
    
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        if '=====================' in line or 'torch.Size' in line:
            continue
        
        line = re.sub(r'<\|ref\|>[^<]*<\|/ref\|><\|det\|>\[\[[^\]]*\]\]<\|/det\|>\s*', '', line)
        line = re.sub(r'<\|[^|]+\|>', '', line)
        
        if '<table>' in line and '</table>' in line:
            markdown_table = html_table_to_markdown(line)
            processed_lines.append(markdown_table)
        else:
            processed_lines.append(line)
    
    cleaned_lines = []
    for line in processed_lines:
        line = line.strip()
        if line or (cleaned_lines and cleaned_lines[-1]):
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines)
    result = re.sub(r'\n\n\n+', '\n\n', result)
    
    return result.strip()


class OCRService:
    """Service for running OCR inference using DeepSeek-OCR model with transformers."""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_name = 'deepseek-ai/DeepSeek-OCR'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self):
        """Initialize DeepSeek-OCR with transformers."""
        if self.model is not None:
            return
        
        print(f"Loading DeepSeek-OCR with transformers on {self.device}...")
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else 'cpu',
            trust_remote_code=True
        )
        
        print("DeepSeek-OCR loaded successfully")
    
    def process_image(
        self, 
        image: Image.Image,
        prompt: Optional[str] = None,
        prompt_type: PromptType = PromptType.DOCUMENT,
        mode: OCRMode = OCRMode.GUNDAM
    ) -> str:
        """Process a single image through OCR."""
        if self.model is None:
            self.load_model()
        
        if prompt is None:
            prompt = PROMPT_TEMPLATES[prompt_type]
        
        # Prepare inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=8192,
                temperature=0.0,
                do_sample=False,
            )
        
        # Decode output
        text = self.processor.batch_decode(
            outputs,
            skip_special_tokens=False
        )[0]
        
        return clean_ocr_output(text)
    
    def process_images(
        self, 
        images: List[Image.Image],
        prompt: Optional[str] = None,
        prompt_type: PromptType = PromptType.DOCUMENT,
        mode: OCRMode = OCRMode.GUNDAM
    ) -> List[str]:
        """Process multiple images through OCR."""
        if self.model is None:
            self.load_model()
        
        results = []
        for idx, image in enumerate(images):
            print(f"Processing image {idx + 1}/{len(images)}...")
            result = self.process_image(image, prompt, prompt_type, mode)
            results.append(result)
        
        return results

