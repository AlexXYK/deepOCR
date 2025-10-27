"""
OCR service for DeepSeek-OCR model processing.
"""
import os
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from typing import List, Tuple


class OCRService:
    """
    Service for running OCR inference using DeepSeek-OCR model.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = 'deepseek-ai/DeepSeek-OCR'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self):
        """
        Load the DeepSeek-OCR model and tokenizer.
        Model is loaded with bfloat16 precision and flash attention.
        """
        if self.model is not None:
            return  # Already loaded
        
        print(f"Loading DeepSeek-OCR model on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        
        # Load model with flash attention
        self.model = AutoModel.from_pretrained(
            self.model_name,
            _attn_implementation='flash_attention_2',
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16
        )
        
        # Move to GPU and set to eval mode
        self.model = self.model.eval().to(self.device)
        
        print("Model loaded successfully")
    
    def process_image(
        self, 
        image: Image.Image, 
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown."
    ) -> str:
        """
        Process a single image through OCR.
        
        Args:
            image: PIL Image to process
            prompt: Prompt template for the model
            
        Returns:
            Extracted text as markdown
        """
        if self.model is None:
            self.load_model()
        
        # Run inference
        # Gundam mode: base_size=1024, image_size=640, crop_mode=True
        # Optimizes quality vs speed for large documents
        result = self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image=image,
            output_path='',  # No file output
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,
            test_compress=False
        )
        
        return result
    
    def process_images(
        self, 
        images: List[Image.Image],
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown."
    ) -> List[str]:
        """
        Process multiple images through OCR.
        
        Args:
            images: List of PIL Images to process
            prompt: Prompt template for the model
            
        Returns:
            List of extracted text as markdown
        """
        if self.model is None:
            self.load_model()
        
        results = []
        for idx, image in enumerate(images):
            print(f"Processing image {idx + 1}/{len(images)}...")
            result = self.process_image(image, prompt)
            results.append(result)
        
        return results

