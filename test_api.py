"""
Test script for DeepSeek OCR API.
Usage: python test_api.py
"""
import requests
import json
import sys
from pathlib import Path


API_URL = "http://localhost:5010"


def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        print(f"✓ Health check passed: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_ocr(file_path: str):
    """Test OCR endpoint with a file."""
    print(f"\nTesting OCR with file: {file_path}")
    
    if not Path(file_path).exists():
        print(f"✗ File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f)}
            response = requests.post(
                f"{API_URL}/ocr",
                files=files,
                timeout=300  # OCR can take a while
            )
            response.raise_for_status()
            result = response.json()
            
            print(f"✓ OCR completed successfully")
            print(f"  Status: {result.get('status')}")
            print(f"  Pages: {result.get('pages')}")
            print(f"  File type: {result.get('file_type')}")
            print(f"  Text length: {len(result.get('text', ''))} characters")
            
            # Show first 200 characters of output
            text_preview = result.get('text', '')[:200]
            if text_preview:
                print(f"\n  Preview:\n{text_preview}...")
            
            return True
            
    except requests.Timeout:
        print("✗ Request timed out")
        return False
    except Exception as e:
        print(f"✗ OCR failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DeepSeek OCR API Test Suite")
    print("=" * 60)
    
    # Test health
    if not test_health():
        print("\n❌ Health check failed. Is the server running?")
        sys.exit(1)
    
    # Test OCR if file provided
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        test_ocr(test_file)
    else:
        print("\n💡 Tip: Provide a file path to test OCR:")
        print("  python test_api.py /path/to/image.jpg")
        print("  python test_api.py /path/to/document.pdf")
    
    print("\n" + "=" * 60)
    print("Test suite completed")
    print("=" * 60)


if __name__ == "__main__":
    main()

