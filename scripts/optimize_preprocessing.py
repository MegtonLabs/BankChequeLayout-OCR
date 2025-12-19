import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os

def load_model():
    print("Loading TrOCR model...")
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    print("Model loaded.")
    return processor, model

def ocr(image, processor, model):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def apply_preprocessing(img, method, block_size=11, C=2, morph=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    if method == 'adaptive_gaussian':
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
    elif method == 'adaptive_mean':
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
    elif method == 'otsu':
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        return None
        
    if morph == 'open_2x2':
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    elif morph == 'close_2x2':
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    elif morph == 'open_3x3':
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def optimize(image_path, processor, model):
    print(f"Optimizing for {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found.")
        return

    methods = ['adaptive_gaussian', 'adaptive_mean', 'otsu']
    block_sizes = [11, 15, 21, 31, 41]
    Cs = [2, 5, 10, 15]
    morphs = [None, 'open_2x2', 'close_2x2']
    
    results = []
    
    for method in methods:
        if method == 'otsu':
            # Otsu doesn't use block_size or C
            processed = apply_preprocessing(img, method)
            pil_img = Image.fromarray(processed)
            text = ocr(pil_img, processor, model)
            print(f"Method: {method}, Text: {text}")
            results.append((text, f"{method}"))
            continue
            
        for bs in block_sizes:
            for c in Cs:
                for m in morphs:
                    processed = apply_preprocessing(img, method, bs, c, m)
                    pil_img = Image.fromarray(processed)
                    text = ocr(pil_img, processor, model)
                    config = f"{method}, BS={bs}, C={c}, Morph={m}"
                    print(f"Config: {config} -> Text: {text}")
                    results.append((text, config))
                    
    return results

if __name__ == "__main__":
    processor, model = load_model()
    
    print("-" * 30)
    print("Testing Date...")
    optimize('../fields/Date.jpg', processor, model)
    
    print("-" * 30)
    print("Testing Amount...")
    optimize('../fields/padded_amount.jpg', processor, model)
