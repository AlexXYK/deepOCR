"""
OCR Service using transformers library for DeepSeek-OCR
Production-ready implementation for commercialization
"""

from typing import Optional
from PIL import Image
from enum import Enum
import torch


class OCRMode(str, Enum):
    """OCR processing modes"""
    GUNDAM = "gundam"  # Original DeepSeek-OCR mode
    PIXTRAL = "pixtral"  # Alternate mode


class PromptType(str, Enum):
    """Pre-defined prompt types for common OCR tasks"""
    DOCUMENT = "document"
    TABLE = "table"
    FORMULA = "formula"
    HANDWRITING = "handwriting"


PROMPT_TEMPLATES = {
    PromptType.DOCUMENT: "Extract all text from this document, preserving formatting and structure.",
    PromptType.TABLE: "Extract the table data from this image, maintaining its structure.",
    PromptType.FORMULA: "Extract all mathematical formulas and equations from this image.",
    PromptType.HANDWRITING: "Extract all handwritten text from this image.",
}


def clean_ocr_output(text: str) -> str:
    """Clean up OCR output by removing special tokens"""
    # Remove common special tokens
    special_tokens = ["<|begin_of_text|>", "<|end_of_text|>", "<|im_start|>", "<|im_end|>"]
    for token in special_tokens:
        text = text.replace(token, "")
    return text.strip()


class OCRService:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_name = 'deepseek-ai/DeepSeek-OCR'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        """Load DeepSeek-OCR model using transformers"""
        if self.model is not None:
            return

        print(f"Loading DeepSeek-OCR with transformers on {self.device}...")
        
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device == 'cuda' else torch.float32,
            device_map="auto" if self.device == 'cuda' else None,
            attn_implementation="flash_attention_2" if self.device == 'cuda' else None,
        )
        
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        print("Model loaded successfully")

    def process_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        prompt_type: PromptType = PromptType.DOCUMENT,
        mode: OCRMode = OCRMode.GUNDAM
    ) -> str:
        """
        Process an image with OCR
        
        Args:
            image: PIL Image to process
            prompt: Custom prompt (if None, uses prompt_type template)
            prompt_type: Type of OCR task
            mode: Processing mode
            
        Returns:
            Extracted text
        """
        if self.model is None:
            self.load_model()

        if prompt is None:
            prompt = PROMPT_TEMPLATES[prompt_type]

        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process with processor
        inputs = self.processor(
            messages=messages,
            images=[image],
            return_tensors="pt"
        ).to(self.device)
        
        # Generate with optimal parameters for OCR
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=8192,
                do_sample=False,  # Deterministic for OCR
                temperature=0.0,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        # Decode output
        text = self.processor.decode(
            outputs[0],
            skip_special_tokens=False  # We'll clean them manually
        )
        
        return clean_ocr_output(text)

