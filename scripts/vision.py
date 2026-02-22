from imports import *
from PIL import Image
import numpy as np
import cv2
import base64
import requests
import json
import io

print("Initializing GLM-OCR from Ollama...")

# Configuration for Ollama GLM-OCR
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "glm-ocr"  # Make sure this model is available in Ollama

print(f"GLM-OCR will use Ollama model: {OLLAMA_MODEL}")
print("Note: Make sure Ollama is running and the glm-ocr model is installed")
print("Run: ollama pull glm-ocr (if not already installed)")

def vision_api(f):
    """
    Extract text from image using Ollama GLM-OCR model
    
    Args:
        f: Image file path (str), numpy array (OpenCV format), or PIL Image
        
    Returns:
        List containing extracted text
    """
    try:
        # Convert input to PIL Image
        if isinstance(f, str):
            img = Image.open(f).convert("RGB")
        elif isinstance(f, np.ndarray):
            # Convert OpenCV (BGR) to PIL (RGB)
            img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        else:
            # Assuming f is already a PIL Image or similar
            img = f.convert("RGB")

        # Preprocessing for better OCR
        # 1. Resize if too small (GLM-OCR works better with reasonable-sized images)
        if img.width < 384 or img.height < 384:
            scale = max(384 / img.width, 384 / img.height)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # 2. Enhance contrast
        img_np = np.array(img)
        # Convert to LAB color space
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        L_channel, a_channel, b_channel = cv2.split(lab)
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        L_enhanced = clahe.apply(L_channel)
        # Merge channels
        lab_enhanced = cv2.merge((L_enhanced, a_channel, b_channel))
        # Convert back to RGB
        img_np = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        img = Image.fromarray(img_np)

        # Convert image to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Prepare prompt for GLM-OCR
        prompt = "Extract and recognize all text from this image. Return only the extracted text, nothing else."

        # Call Ollama API
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [img_base64],
            "stream": False,
            "temperature": 0.3,  # Lower temperature for more consistent OCR
        }

        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "").strip()
            
            if generated_text:
                # Return as a list to match previous interface
                return [generated_text]
            else:
                print("GLM-OCR returned empty response")
                return []
        else:
            print(f"Ollama API error: {response.status_code} - {response.text}")
            return []

    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to Ollama. Make sure:")
        print("  1. Ollama is running (ollama serve)")
        print("  2. The glm-ocr model is installed (ollama pull glm-ocr)")
        print("  3. Ollama is accessible at http://localhost:11434")
        return []
    except Exception as e:
        print(f"GLM-OCR Error: {e}")
        import traceback
        traceback.print_exc()
        return []