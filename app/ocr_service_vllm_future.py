"""
OCR service for DeepSeek-OCR model processing.
"""
import os
import torch
import re
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional
from enum import Enum
from vllm import LLM, SamplingParams
try:
    from bs4 import BeautifulSoup
    BEAUTIFUL_SOUP_AVAILABLE = True
except ImportError:
    BEAUTIFUL_SOUP_AVAILABLE = False


class OCRMode(str, Enum):
    """OCR mode configurations."""
    TINY = "tiny"           # 512x512, 64 tokens
    SMALL = "small"         # 640x640, 100 tokens
    BASE = "base"          # 1024x1024, 256 tokens
    LARGE = "large"        # 1280x1280, 400 tokens
    GUNDAM = "gundam"      # n×640×640 + 1×1024×1024 (optimal for documents)


class PromptType(str, Enum):
    """Predefined prompt types for different OCR tasks."""
    DOCUMENT = "document"        # Markdown with layout preservation
    FREE_OCR = "free_ocr"       # Free OCR without layout
    FIGURES = "figures"         # Parse figures in documents
    DESCRIBE = "describe"       # Describe image in detail
    OTHER = "other"             # General images


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
        # If BeautifulSoup not available, return as-is
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
        
        # Convert to markdown table
        max_cols = max(len(row) for row in rows)
        
        # Pad rows to max_cols
        for row in rows:
            while len(row) < max_cols:
                row.append('')
        
        # Create markdown table
        markdown_lines = []
        markdown_lines.append('| ' + ' | '.join(rows[0]) + ' |')
        markdown_lines.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
        
        for row in rows[1:]:
            markdown_lines.append('| ' + ' | '.join(row) + ' |')
        
        return '\n'.join(markdown_lines)
    except Exception as e:
        print(f"Error converting table: {e}")
        return html


def add_smart_line_breaks(text: str) -> str:
    """
    Add intelligent line breaks to structured text.
    Conservative approach - only break on clear structure patterns.
    """
    if not text:
        return ""
    
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Skip if line is already a table, heading, or separator
        if line.strip().startswith('|') or line.strip().startswith('#') or line.strip().startswith('---'):
            formatted_lines.append(line)
            continue
        
        # Don't break lines that are already multiline
        if '\n' in line:
            formatted_lines.append(line)
            continue
        
        # Don't break phone numbers
        if re.search(r'\(?\d{3}\)?\s*-?\s*\d{3}\s*-?\s*\d{4}', line):
            formatted_lines.append(line)
            continue
        
        # Very conservative: only break on clear list-like patterns with multiple colons
        # that suggest multiple fields (e.g., "Species: Canine Sex: Male Age: 1 year")
        if line.count(':') > 1 and not 'http' in line.lower():
            # Split on " Label:" pattern to separate fields
            parts = re.split(r'(\s+[A-Z][a-z]+\s*:)', line)
            if len(parts) > 2:
                # Found multiple field labels
                result = []
                for i in range(len(parts)):
                    result.append(parts[i])
                    # Add line break after label if followed by value
                    if i < len(parts) - 1 and parts[i].strip().endswith(':'):
                        if parts[i+1] and not parts[i+1].strip().startswith(':'):
                            result.append('\n')
                formatted_lines.append(''.join(result))
            else:
                formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)


def clean_ocr_output(text: str) -> str:
    """
    Clean OCR output by removing metadata tags and formatting markers.
    
    Removes:
    - <|ref|>tags<|/ref|> and <|det|>[[coords]]<|/det|> markers
    - Debug information like "====================="
    - Empty lines and extra whitespace
    
    Also converts HTML tables to markdown tables.
    """
    if not text:
        return ""
    
    lines = text.split('\n')
    
    # First pass: extract and convert HTML tables
    processed_lines = []
    current_table = []
    in_table = False
    
    for line in lines:
        # Skip debug/separator lines
        if '=====================' in line or 'torch.Size' in line:
            continue
        
        # Remove <|ref|>tag<|/ref|><|det|>[[coords]]<|/det|> patterns
        line = re.sub(r'<\|ref\|>[^<]*<\|/ref\|><\|det\|>\[\[[^\]]*\]\]<\|/det\|>\s*', '', line)
        
        # Remove any remaining <|ref|> or <|det|> tags
        line = re.sub(r'<\|[^|]+\|>', '', line)
        
        # Check for complete HTML table in a single line
        if '<table>' in line and '</table>' in line:
            markdown_table = html_table_to_markdown(line)
            processed_lines.append(markdown_table)
        elif '<table>' in line:
            in_table = True
            current_table = [line]
        elif '</table>' in line:
            current_table.append(line)
            in_table = False
            table_html = '\n'.join(current_table)
            markdown_table = html_table_to_markdown(table_html)
            processed_lines.append(markdown_table)
            current_table = []
        elif in_table:
            current_table.append(line)
        else:
            processed_lines.append(line)
    
    # Second pass: clean up whitespace
    cleaned_lines = []
    
    for line in processed_lines:
        # Skip debug messages
        if any(debug_msg in line for debug_msg in ['Trying with file path', 'Captured output', 'Model returned', 'Error converting', '/opt/conda/lib/python3', 'UserWarning', 'warnings.warn']):
            continue
        
        # Clean up extra whitespace
        line = line.strip()
        
        # Skip completely empty lines unless the last line wasn't empty (preserve structure)
        if line or (cleaned_lines and cleaned_lines[-1]):
            cleaned_lines.append(line)
    
    # Join lines and clean up multiple consecutive newlines
    result = '\n'.join(cleaned_lines)
    result = re.sub(r'\n\n\n+', '\n\n', result)
    
    # Apply smart line breaking
    result = add_smart_line_breaks(result)
    
    return result.strip()


class OCRService:
    """
    Service for running OCR inference using DeepSeek-OCR model.
    """
    
    def __init__(self):
        self.llm = None
        self.model_name = 'deepseek-ai/DeepSeek-OCR'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self):
        """
        Initialize vLLM LLM for DeepSeek-OCR.
        """
        if self.llm is not None:
            return  # Already loaded
        
        print(f"Loading DeepSeek-OCR vLLM on {self.device}...")
        
        # Import NGramPerReqLogitsProcessor for DeepSeek-OCR
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
        
        # Initialize vLLM with DeepSeek-OCR specific configuration
        self.llm = LLM(
            model=self.model_name,
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,  # Use 90% of GPU memory
            max_model_len=8192,
            logits_processors=[NGramPerReqLogitsProcessor],  # Required for DeepSeek-OCR
        )
        print("vLLM initialized successfully")
    
    def _get_mode_params(self, mode: OCRMode) -> dict:
        """Get parameters for a specific OCR mode."""
        modes = {
            OCRMode.TINY: {"base_size": 512, "image_size": 512, "crop_mode": False},
            OCRMode.SMALL: {"base_size": 640, "image_size": 640, "crop_mode": False},
            OCRMode.BASE: {"base_size": 1024, "image_size": 1024, "crop_mode": False},
            OCRMode.LARGE: {"base_size": 1280, "image_size": 1280, "crop_mode": False},
            OCRMode.GUNDAM: {"base_size": 1024, "image_size": 640, "crop_mode": True},
        }
        return modes.get(mode, modes[OCRMode.GUNDAM])
    
    def process_image(
        self, 
        image: Image.Image,
        prompt: Optional[str] = None,
        prompt_type: PromptType = PromptType.DOCUMENT,
        mode: OCRMode = OCRMode.GUNDAM
    ) -> str:
        """
        Process a single image through OCR.
        
        Args:
            image: PIL Image to process
            prompt: Custom prompt (overrides prompt_type if provided)
            prompt_type: Predefined prompt type (ignored if prompt is provided)
            mode: OCR mode (kept for API compatibility; vLLM handles image processing internally)
            
        Returns:
            Extracted text as markdown
        """
        if self.llm is None:
            self.load_model()
        
        # Determine prompt to use
        if prompt is None:
            prompt = PROMPT_TEMPLATES[prompt_type]
        
        # Assemble vLLM inputs
        model_input = {
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        }
        
        # DeepSeek-OCR sampling configuration with NGram parameters
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            skip_special_tokens=False,
            # NGram logits processor configuration
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
            ),
        )
        
        # Run generation
        outputs = self.llm.generate([model_input], sampling_params)
        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        return clean_ocr_output(text)
    
    def process_images(
        self, 
        images: List[Image.Image],
        prompt: Optional[str] = None,
        prompt_type: PromptType = PromptType.DOCUMENT,
        mode: OCRMode = OCRMode.GUNDAM
    ) -> List[str]:
        """
        Process multiple images through OCR.
        
        Args:
            images: List of PIL Images to process
            prompt: Custom prompt (overrides prompt_type if provided)
            prompt_type: Predefined prompt type (ignored if prompt is provided)
            mode: OCR mode configuration
            
        Returns:
            List of extracted text as markdown
        """
        if self.model is None:
            self.load_model()
        
        results = []
        for idx, image in enumerate(images):
            print(f"Processing image {idx + 1}/{len(images)}...")
            result = self.process_image(image, prompt, prompt_type, mode)
            results.append(result)
        
        return results

