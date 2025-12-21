from imports import *
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, logging
import torch
from PIL import Image
import numpy as np
import cv2

# Suppress warnings
logging.set_verbosity_error()

import os
print("Loading TrOCR model (./models/trocr-base-printed)... this may take a moment.")
# Get the absolute path to the project root (assuming scripts/vision.py structure)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
model_path = os.path.join(project_root, 'models', 'trocr-base-printed')

# Check if model exists locally; if not, download and save it
if not os.path.exists(model_path) or not any(fname.endswith('.json') for fname in os.listdir(model_path)):
    print(f"Model not found at {model_path}. Downloading from Hugging Face...")
    
    # Use a temporary cache dir inside the project to avoid touching system global cache
    temp_cache_dir = os.path.join(model_path, "temp_cache")
    if not os.path.exists(temp_cache_dir):
        os.makedirs(temp_cache_dir, exist_ok=True)
        
    print(f"Downloading to temporary cache: {temp_cache_dir}...")
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed', cache_dir=temp_cache_dir)
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed', cache_dir=temp_cache_dir)
    
    print(f"Saving model to {model_path}...")
    processor.save_pretrained(model_path)
    model.save_pretrained(model_path)
    
    # Cleanup temp cache
    print("Cleaning up temporary cache...")
    import shutil
    if os.path.exists(temp_cache_dir):
        shutil.rmtree(temp_cache_dir)
else:
    print(f"Loading from local path: {model_path}")
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)

print("TrOCR model loaded.")

def vision_api(f):
    try:
        # Read image
        if isinstance(f, str):
            img = Image.open(f).convert("RGB")
        elif isinstance(f, np.ndarray):
            # Convert OpenCV (BGR) to PIL (RGB)
            img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        else:
            # Assuming f is already a PIL Image or similar
            img = f.convert("RGB")

        # Preprocessing for better OCR
        # 1. Resize if too small (TrOCR works better with larger text)
        if img.width < 384 or img.height < 384:
            scale = max(384/img.width, 384/img.height)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # 2. Enhance contrast (optional, but often helps with faint text)
        # Convert to numpy for OpenCV processing
        img_np = np.array(img)
        # Convert to LAB color space
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        # Merge channels
        limg = cv2.merge((cl,a,b))
        # Convert back to RGB
        img_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        img = Image.fromarray(img_np)

        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Return as a list of words/strings to match previous interface
        # The previous interface returned a list of words.
        # TrOCR returns a full sentence/line. 
        # We can return a list containing just the full text, or split it.
        # Given the usage in main.py: extracted_text = " ".join(extracted_text_list).strip()
        # Returning [generated_text] is safe.
        return [generated_text]

    except Exception as e:
        print(f"TrOCR Error: {e}")
        return []