import os
# Force transformers to use PyTorch to avoid broken TensorFlow installation (DLL error)
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from imports import *
from vision import vision_api
import argparse
from ultralytics import YOLO
import pandas as pd
import cv2
import os
import base64
import requests

import re
SIGNATURE_MODEL = "glm-ocr"

def validate_signature(crop):
    """
    Use Ollama vision model to check if a handwritten signature exists.
    Falls back to CV pixel analysis if Ollama fails.
    """
    if crop is None or crop.size == 0:
        print("    [DEBUG] Signature crop is empty/None")
        return False

    cv2.imwrite("../fields/debug_signature_crop.jpg", crop)

    # ── Step 1: Detect available Ollama model ────────────────────────
    model_name = _detect_ollama_model()
    if not model_name:
        print("    [WARNING] No Ollama model found. Falling back to CV check.")
        return _cv_fallback(crop)

    # ── Step 2: Encode image ─────────────────────────────────────────
    _, buf = cv2.imencode(".jpg", crop)
    img_b64 = base64.b64encode(buf).decode("utf-8")

    # ── Step 3: Ask Ollama ───────────────────────────────────────────
    answer = _ask_ollama(model_name, img_b64)

    if answer is not None:
        return answer

    # ── Step 4: Fallback to CV ───────────────────────────────────────
    print("    [DEBUG] Ollama gave no clear answer. Falling back to CV check.")
    return _cv_fallback(crop)


def _detect_ollama_model():
    """Auto-detect the first available Ollama model."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            print(f"    [DEBUG] Ollama models found: {model_names}")

            if model_names:
                # Prefer vision/VQA models over pure-text ones
                vqa_preferred = ["llava", "minicpm", "bakllava", "glm"]
                for pref in vqa_preferred:
                    for m in model_names:
                        if pref in m.lower():
                            print(f"    [DEBUG] Selected model: {m}")
                            return m
                # Fallback: use first available
                print(f"    [DEBUG] Selected model (fallback): {model_names[0]}")
                return model_names[0]
    except requests.exceptions.ConnectionError:
        print("    [ERROR] Cannot connect to Ollama at localhost:11434")
        print("    [ERROR] Make sure Ollama is running: ollama serve")
    except Exception as e:
        print(f"    [ERROR] Model detection failed: {e}")
    return None


def _ask_ollama(model_name, img_b64):
    """
    Try BOTH Ollama API formats (chat and generate).
    Returns True/False or None if unclear.
    """
    prompt = (
        "Look at this image. It is cropped from the signature area of a bank cheque.\n"
        "Does this image contain a HANDWRITTEN signature made by a person with a pen?\n\n"
        "IMPORTANT RULES:\n"
        "- Printed text like 'Authorised Signatory' is NOT a signature → answer NO\n"
        "- Straight lines or empty space is NOT a signature → answer NO\n"
        "- Only actual handwritten pen/ink strokes count as a signature → answer YES\n\n"
        "Answer with ONLY one word: YES or NO"
    )

    # ── Try /api/chat (works with most vision models) ────────────────
    answer = _try_chat_api(model_name, prompt, img_b64)
    if answer is not None:
        return answer

    # ── Try /api/generate (older format) ─────────────────────────────
    answer = _try_generate_api(model_name, prompt, img_b64)
    if answer is not None:
        return answer

    return None


def _try_chat_api(model_name, prompt, img_b64):
    """Try Ollama /api/chat endpoint."""
    try:
        print(f"    [DEBUG] Trying /api/chat with model={model_name}...")
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_b64],
                    }
                ],
                "stream": False,
            },
            timeout=120,
        )
        print(f"    [DEBUG] /api/chat status: {resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            print(f"    [DEBUG] /api/chat raw response: '{content}'")
            return _parse_yes_no(content)
        else:
            print(f"    [DEBUG] /api/chat error body: {resp.text[:200]}")
    except requests.exceptions.ConnectionError:
        print("    [ERROR] Ollama not running")
    except Exception as e:
        print(f"    [DEBUG] /api/chat exception: {e}")
    return None


def _try_generate_api(model_name, prompt, img_b64):
    """Try Ollama /api/generate endpoint."""
    try:
        print(f"    [DEBUG] Trying /api/generate with model={model_name}...")
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
            },
            timeout=120,
        )
        print(f"    [DEBUG] /api/generate status: {resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            content = data.get("response", "").strip()
            print(f"    [DEBUG] /api/generate raw response: '{content}'")
            return _parse_yes_no(content)
        else:
            print(f"    [DEBUG] /api/generate error body: {resp.text[:200]}")
    except requests.exceptions.ConnectionError:
        print("    [ERROR] Ollama not running")
    except Exception as e:
        print(f"    [DEBUG] /api/generate exception: {e}")
    return None


def _parse_yes_no(text):
    """
    Parse LLM response to extract YES/NO answer.
    Returns True, False, or None (ambiguous).
    """
    if not text:
        return None

    text_lower = text.lower().strip()
    words = re.sub(r"[^a-z\s]", " ", text_lower).split()

    print(f"    [DEBUG] Parsed words: {words[:10]}")

    # Direct match
    if words and words[0] in ("yes", "no"):
        result = words[0] == "yes"
        print(f"    [DEBUG] Direct match: {result}")
        return result

    # Check for negation patterns (no signature, not present, etc.)
    neg_phrases = [
        "no signature", "no handwritten", "not contain", "does not",
        "doesn't", "don't see", "cannot see", "no pen", "no ink",
        "is not", "isn't", "empty", "blank", "only printed",
        "no there", "there is no"
    ]
    for phrase in neg_phrases:
        if phrase in text_lower:
            print(f"    [DEBUG] Negative phrase match: '{phrase}'")
            return False

    # Check for affirmative patterns
    pos_phrases = [
        "yes", "signature present", "handwritten signature",
        "contains a signature", "there is a signature",
        "signature is present", "signed"
    ]
    for phrase in pos_phrases:
        if phrase in text_lower:
            print(f"    [DEBUG] Positive phrase match: '{phrase}'")
            return True

    print(f"    [DEBUG] Could not parse YES/NO from: '{text[:100]}'")
    return None


def _cv_fallback(crop):
    """
    Simple CV-based check: measure ink pixels after removing lines.
    Only used when Ollama is unavailable.
    """
    print("    [DEBUG] Running CV fallback...")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape

    # Remove horizontal lines
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 3, 40), 2))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_k, iterations=2)
    h_lines = cv2.dilate(h_lines,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)),
                         iterations=1)

    # Remove vertical lines
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, max(h // 3, 40)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_k, iterations=2)
    v_lines = cv2.dilate(v_lines,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
                         iterations=1)

    # Subtract lines
    all_lines = cv2.bitwise_or(h_lines, v_lines)
    clean = cv2.bitwise_and(binary, binary, mask=cv2.bitwise_not(all_lines))

    # Remove small noise
    noise_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, noise_k)

    cv2.imwrite("../fields/debug_sig_cv_clean.jpg", clean)

    ink = cv2.countNonZero(clean)
    total = h * w if h * w > 0 else 1
    ratio = ink / total

    print(f"    [DEBUG] CV fallback: ink={ink}, total={total}, ratio={ratio:.4f}")

    # Threshold: real signatures typically cover 0.5-5% of crop area
    return ratio > 0.005

def preprocess_image(crop, field_type, method='default'):
    """
    Apply specific image processing based on field type to improve OCR.
    """
    if crop is None or crop.size == 0:
        return crop
        
    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    if field_type == 'Date':
        # Date often has boxes or lines. 
        # Increase contrast and threshold
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Binarize to remove faint background noise
        if method == 'adaptive_mean':
             # Better for noisy/grainy images like Cheque_96
             binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                            cv2.THRESH_BINARY, 21, 5)
        else:
             # Default: Gaussian (better for uneven illumination/shadows)
             binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 15, 5)
                                       
        # Remove vertical lines (box separators)
        # Increase kernel height to avoid removing handwritten vertical strokes
        # Crop height is ~100px. Use 80px to only catch full height lines.
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
        detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        
        # Check if lines are thin (box separators) or thick (handwriting)
        # Erode horizontally. Thin lines (1-4px) will disappear. Thick strokes will survive.
        horizontal_erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        surviving_lines = cv2.erode(detected_lines, horizontal_erode_kernel, iterations=1)
        
        surviving_area = cv2.countNonZero(surviving_lines)
        total_line_area = cv2.countNonZero(detected_lines)
        
        # If surviving area is small (< 20% of total line area), it's mostly thin lines -> Remove them.
        # If surviving area is large, it's thick strokes -> Keep them (don't remove).
        
        if total_line_area > 0 and (surviving_area / total_line_area) < 0.2:
             # print("    [DEBUG] Removing thin vertical lines.")
             # Dilate lines to ensure full removal
             detected_lines = cv2.dilate(detected_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
             
             # Mask out lines
             binary_inv = cv2.bitwise_not(binary)
             binary_inv_no_lines = cv2.bitwise_and(binary_inv, binary_inv, mask=cv2.bitwise_not(detected_lines))
             binary_clean = cv2.bitwise_not(binary_inv_no_lines)
        else:
             # print("    [DEBUG] Keeping vertical strokes (likely handwriting).")
             binary_clean = binary
        
        # Morphological opening to remove small noise (dots)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel) 
        
        # binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel)
                                       
        # Save debug image
        cv2.imwrite('../fields/debug_date.jpg', binary_clean)
        
        return cv2.cvtColor(binary_clean, cv2.COLOR_GRAY2BGR)
        
    elif field_type == 'IFSC':
        # Sharpening might help
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
    elif field_type == 'Amount':
        # Amount often has background patterns or is faint.
        # Apply CLAHE to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Binarize to remove background
        # Use Adaptive Thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 15, 5)
        
        # Optional: Morphological opening to remove small noise (dots)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Save debug image
        cv2.imwrite('../fields/debug_amount.jpg', binary)
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
    return crop

def clean_text(text, field_type):
    if not text:
        return ""
    
    text = text.strip()
    
    if field_type == 'BankName':
        # Look for known bank names if the OCR is noisy
        known_banks = [
            "SyndicateBank", "State Bank of India", "HDFC Bank", "ICICI Bank", 
            "Axis Bank", "Punjab National Bank", "Canara Bank", "Bank of Baroda",
            "Union Bank of India", "IDBI Bank", "Indian Bank"
        ]
        
        # Check for keywords to isolate the specific bank name
        for bank in known_banks:
            if bank.lower() in text.lower():
                return bank
                
        # If no strict match, try to clean common noise
        # Remove "भारत सरकार का उपक्रम", "Govt. of India Undertaking", etc.
        noise_patterns = [
            r'भारत सरकार.*',
            r'Govt\. of India.*',
            r'A Govt\. of India.*',
            r'Understaking',
            r'विश्वसनीय.*',
            r'Faithful.*',
            r'Friendly.*'
        ]
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
            
        return text.strip()
    
    if field_type == 'IFSC':
        # Remove common noise words
        # FIRISC, TAX, TEL, MAS, etc might appear nearby
        text = re.sub(r'(IFSC|IFS|ITEMS|CODE|BANK|NO|FIRISC|TAX|TEL|MAS|[:\s-])', '', text, flags=re.IGNORECASE)
        # Keep only alphanumeric
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Search for pattern first
        # Standard IFSC format: 4 letters, 0, 6 alphanumeric characters
        # Regex: ^[A-Z]{4}0[A-Z0-9]{6}$
        match = re.search(r'([A-Z]{4}0[A-Z0-9]{6})', text)
        if match:
            found_code = match.group(1)
            # Fix typos in the extracted code
            if found_code.startswith('UTHB'):
                found_code = found_code.replace('UTHB', 'UTIB', 1)
            if found_code.startswith('1CIC'):
                found_code = found_code.replace('1CIC', 'ICIC', 1)
            return found_code
        
        # If no strict match, try to fix typos in the whole text then search again?
        # Or just apply fixes to text and check length
        
        # Common OCR typos in Bank Codes
        # UTHB -> UTIB (Axis)
        if 'UTHB' in text:
            text = text.replace('UTHB', 'UTIB')
        # 1CIC -> ICIC
        if '1CIC' in text:
            text = text.replace('1CIC', 'ICIC')
            
        # Try matching again after fix
        match = re.search(r'([A-Z]{4}0[A-Z0-9]{6})', text)
        if match:
            return match.group(1)
            
        # Fallback: Try to fix common errors
        if len(text) == 11:
            text_list = list(text)
            # 5th char should be 0
            if text_list[4] in ['O', 'D', 'Q', 'C']: 
                text_list[4] = '0'
            return "".join(text_list)
            
        return text
        
    elif field_type == 'Amount':
        # Fix common OCR errors BEFORE removing non-digits
        text = text.replace('NO', '10')
        text = text.replace('No', '10')
        text = text.replace('O', '0').replace('o', '0')
        
        # Remove currency symbols and text, but keep spaces for now to detect separation?
        # Actually, spaces usually mean nothing in amount unless it's "25 000".
        # Let's remove spaces.
        text = re.sub(r'[^\d,.]', '', text)
        
        # Fix common OCR errors
        text = text.replace('ID', '10')
        
        # Fix trailing '1' which is often '/-' misread
        # Example: "39,0001" -> "39,000"
        if text.endswith('1') and len(text) > 1:
            if text.endswith('001'):
                text = text[:-1]
            elif text.endswith(',001'): # 25,001 -> 25,000? Risky.
                pass
            # Heuristic: If the number of digits after the last comma is 4, and the last is 1
            last_comma_index = text.rfind(',')
            if last_comma_index != -1:
                suffix = text[last_comma_index+1:]
                if len(suffix) == 4 and suffix.endswith('1'):
                     text = text[:-1]
        
        # Heuristic: If there are multiple dots
        # e.g. 55.00.00 -> 5500.00
        if text.count('.') > 1:
            parts = text.split('.')
            # If the last part is 00, keep it as decimal
            if parts[-1] == '00':
                text = "".join(parts[:-1]) + '.00'
            else:
                # Otherwise, assume all dots are commas
                text = ",".join(parts)
                
        # Heuristic: If dot is followed by 3 digits (and not end of string), it's likely a comma
        text = re.sub(r'\.(\d{3})', r',\1', text)
        
        # Heuristic: If dot is followed by 2 digits and a comma (e.g. 25.35,000), it's a comma
        text = re.sub(r'\.(\d{2}),', r',\1,', text)
        
        # Fix double commas
        while ',,' in text:
            text = text.replace(',,', ',')
            
        # Format with Indian Numbering System if no commas exist and it's a large number
        if ',' not in text and '.' not in text and len(text) > 3:
            # e.g. 25000000 -> 2,50,00,000
            # Last 3 digits
            last3 = text[-3:]
            rest = text[:-3]
            # Group rest by 2
            if rest:
                rest = re.sub(r'\B(?=(\d{2})+(?!\d))', ",", rest)
                text = f"{rest},{last3}"
            else:
                text = last3
                
        if text:
            return f"₹ {text}/-"
        return text
        
    elif field_type == 'Date':
        # Normalize separators
        # Replace common separators with /
        text = text.replace('@', '/').replace('"', '/').replace("'", '/').replace('.', '/').replace('-', '/')
        text = text.replace('\\', '/').replace('|', '/')
        
        # Remove common labels
        text = re.sub(r'(GST|DATE|VALID|UPTO|ISSUE|DD|MM|YY|YYYY)', '', text, flags=re.IGNORECASE)
        
        # Remove letters
        text = re.sub(r'[a-zA-Z]', '', text)
        
        # Handle spaced digits (e.g. 2 5 0 4 2 0 1 5)
        # If we have many spaces between digits, remove them
        if re.search(r'\d\s+\d', text):
            text = text.replace(' ', '')
        
        # Remove everything except digits and /
        text = re.sub(r'[^\d/]', '', text)
        
        # Fix double slashes
        while '//' in text:
            text = text.replace('//', '/')
            
        # Scenario: "25012016" (8 digits) -> Convert to 25/01/2016 then validate
        if len(text) == 8 and text.isdigit():
            text = f"{text[:2]}/{text[2:4]}/{text[4:]}"
            
        # Scenario: "250116" (6 digits) -> Convert to 25/01/2016 then validate
        if len(text) == 6 and text.isdigit():
            text = f"{text[:2]}/{text[2:4]}/20{text[4:]}"
            
        # Validate and Fix Date Parts
        parts = text.split('/')
        if len(parts) == 3:
            d, m, y = parts
            
            # Fix Day
            if d.isdigit():
                di = int(d)
                # If day is single digit and > 3, it might be noise + real day?
                # e.g. "9" from "9.29" where 9 is noise?
                # But "9" is a valid day.
                pass
                
            # Fix Month
            if m.isdigit():
                mi = int(m)
                if mi > 12:
                    # Try to infer correction
                    # 29 -> 09 (2->0)
                    if m == '29': m = '09'
                    elif m == '21': m = '01' # 2->0
                    elif m == '22': m = '02'
                    elif m == '20': m = '10' # 2->1
                    elif m == '41': m = '11' # 4->1
                    elif m == '42': m = '12' # 4->1
                    elif m == '71': m = '11' # 7->1
                    
            # Fix Year
            if y.isdigit():
                if len(y) > 4:
                    # e.g. 210157 -> 2015?
                    # If it starts with 20 or 19, take first 4
                    if y.startswith('20') or y.startswith('19'):
                        y = y[:4]
                    # If it looks like 210157, maybe 2015?
                    # 210157 -> 2 10 15 7 -> 2015? Hard to say.
                    # Let's try to find a valid year in it.
                    match = re.search(r'(20\d{2}|19\d{2})', y)
                    if match:
                        y = match.group(1)
                
                if len(y) == 4:
                    yi = int(y)
                    if yi < 2000:
                        # 1917 -> 2017 (1->2, 9->0) - Common confusion
                        if y.startswith('19'):
                            y = '20' + y[2:]
                        
            return f"{d}/{m}/{y}"
            
        return text
        
    elif field_type == 'Cheque MICR Number':
        text = re.sub(r'[^\w\s]', '', text)
        text = " ".join(text.split())
        return text
        
    elif field_type == 'AC/NO':
        text = re.sub(r'[^\d]', '', text)
        return text
        
    return text

from micr_ocr import MICR_OCR

# Initialize MICR OCR
micr_ocr_engine = MICR_OCR()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', help="Path to cheque image", required=True)
    args = parser.parse_args()

    # Load YOLOv11 model
    model_path = '../weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading YOLOv11 model from {model_path}...")
    model = YOLO(model_path)

    # Load image
    img_path = args.input_image
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    print(f"Processing image: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not read image.")
        return

    # Run inference
    results = model(img)
    
    # Dictionary to store extracted text
    # Desired columns: BankName, AC/NO, IFSC, Amount, Cheque MICR Number, Date, Signature
    cheque_fields = {
        'BankName': '',
        'AC/NO': '',
        'IFSC': '',
        'Amount': '',
        'Cheque MICR Number': '',
        'Date': '',
        'Signature': ''
    }

    # Create a copy for annotation
    annotated_img = img.copy()

    print("\nDetections:")
    
    # Define colors for visualization (BGR)
    class_colors = {
        'Account_Number': (255, 0, 0),      # Blue
        'Amount': (0, 0, 255),              # Red
        'Bank_Name': (0, 140, 255),         # Orange
        'Date': (255, 0, 255),              # Magenta
        'IFSC': (0, 255, 255),              # Yellow
        'MICR': (128, 0, 128),              # Purple
        'Signature': (0, 128, 0)            # Dark Green
    }

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            
            print(f"  - {cls_name} ({conf:.2f})")

            # Get color for this class (default to Green if not found)
            color = class_colors.get(cls_name, (0, 255, 0))

            # Draw bounding box with specific color
            cv2.rectangle(annotated_img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)

            # Add label background and text
            label = f"{cls_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_img, (int(xyxy[0]), int(xyxy[1]) - 20), (int(xyxy[0]) + w, int(xyxy[1])), color, -1)
            cv2.putText(annotated_img, label, (int(xyxy[0]), int(xyxy[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Crop the image with padding
            # Padding helps OCR and contour detection (especially for MICR)
            x1, y1, x2, y2 = xyxy
            
            # Default padding
            padding = 20
            
            # Reduce padding for Date to avoid noise (e.g. lines)
            if 'date' in cls_name.lower():
                padding = 5 # Increased from 0 to avoid cutting off text
                
            h, w = img.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            crop = img[y1:y2, x1:x2]
            
            # Map class to field
            field_key = None
            if 'date' in cls_name.lower():
                field_key = 'Date'
            elif 'amount' in cls_name.lower():
                field_key = 'Amount'
            elif 'ifsc' in cls_name.lower():
                field_key = 'IFSC'
            elif 'account' in cls_name.lower() or 'ac/no' in cls_name.lower():
                field_key = 'AC/NO'
            elif 'micr' in cls_name.lower():
                field_key = 'Cheque MICR Number'
            elif 'bank' in cls_name.lower():
                 field_key = 'BankName'
            elif 'signature' in cls_name.lower():
                 field_key = 'Signature'
            
            if field_key:
                # Save the crop to fields folder with specific names requested by user
                filename = f"{field_key}.jpg"
                if field_key == 'Signature':
                    filename = "org_signature.jpg"
                elif field_key == 'Amount':
                    filename = "padded_amount.jpg"
                elif field_key == 'IFSC':
                    filename = "ifsc.jpg"
                elif field_key == 'BankName':
                    filename = "Bank name.jpg"
                
                # Sanitize filename (replace / with _)
                filename = filename.replace('/', '_')
                
                save_path = f"../fields/{filename}"
                cv2.imwrite(save_path, crop)
                print(f"    Saved crop to {save_path}")

                if field_key == 'Cheque MICR Number':
                    # Use Template Matching for MICR
                    print(f"    Extracting text for {field_key} (Template Matching)...")
                    extracted_text = micr_ocr_engine.predict(crop)
                    print(f"    Result: {extracted_text}")
                    cheque_fields[field_key] = extracted_text
                    
                elif field_key == 'Signature':
                    is_valid_visual = validate_signature(crop)
                    if is_valid_visual:
                        print("    Signature validated via ink analysis. Skipping OCR label check.")
                        cheque_fields['Signature'] = 'True'
                    else:
                        print("    Signature detection rejected (empty/noise).")
                        cheque_fields['Signature'] = 'False'
                else:
                    # Perform OCR with TrOCR for other fields
                    print(f"    Extracting text for {field_key}...")
                    
                    # Preprocess crop for better OCR
                    processed_crop = preprocess_image(crop, field_key)
                    
                    extracted_text_list = vision_api(processed_crop)
                    raw_text = " ".join(extracted_text_list).strip()
                    
                    # Clean text
                    cleaned_text = clean_text(raw_text, field_key)
                    print(f"    Result: {cleaned_text} (Raw: {raw_text})")
                    
                    # Retry logic for Date if result is poor
                    if field_key == 'Date':
                        # Check if result looks valid
                        is_valid_date = False
                        if cleaned_text:
                            # Check for standard date format DD/MM/YYYY
                            if re.match(r'\d{2}/\d{2}/\d{4}', cleaned_text):
                                is_valid_date = True
                            # Or if it has a valid year (20xx or 19xx)
                            elif re.search(r'(20\d{2}|19\d{2})', cleaned_text):
                                is_valid_date = True
                        
                        if not is_valid_date:
                            print(f"    [INFO] Date extraction poor ('{cleaned_text}'). Retrying with alternative preprocessing...")
                            # Assuming 'processor' and 'model' are available in this scope for TrOCR
                            # If not, they would need to be passed or initialized.
                            # For this context, assuming they are globally accessible or passed implicitly.
                            processed_crop_alt = preprocess_image(crop, field_key, method='adaptive_mean')
                            cv2.imwrite(f"../fields/{field_key}_alt.jpg", processed_crop_alt)
                            
                            extracted_text_list_alt = vision_api(processed_crop_alt)
                            generated_text_alt = " ".join(extracted_text_list_alt).strip()
                            cleaned_text_alt = clean_text(generated_text_alt, field_key)
                            
                            print(f"    [INFO] Alternative result: '{cleaned_text_alt}'")
                            # If alternative is valid, use it. Or if it's just longer/better?
                            # If original was invalid, and alt is valid, take alt.
                            # If both invalid, take the one with more digits?
                            
                            alt_valid = False
                            if re.match(r'\d{2}/\d{2}/\d{4}', cleaned_text_alt) or re.search(r'(20\d{2}|19\d{2})', cleaned_text_alt):
                                alt_valid = True
                                
                            if alt_valid:
                                 cleaned_text = cleaned_text_alt
                                 # generated_text = generated_text_alt # This variable is not used later, so no need to update
                            elif len(cleaned_text_alt) > len(cleaned_text):
                                 # Fallback: if both invalid, take the longer one (more info)
                                 cleaned_text = cleaned_text_alt
                                 # generated_text = generated_text_alt # This variable is not used later, so no need to update
                    cheque_fields[field_key] = cleaned_text

    # Save the annotated image
    # Save the annotated image (Legacy debug)
    cv2.imwrite('../fields/check_cont_.jpg', annotated_img)

    # Save to predictions folder
    predictions_dir = '../predictions'
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
        
    base_name = os.path.basename(img_path)
    prediction_path = os.path.join(predictions_dir, base_name)
    cv2.imwrite(prediction_path, annotated_img)
    print(f"Annotated image saved to {prediction_path}")

    # Create DataFrame with specific column order
    columns_order = ['BankName', 'AC/NO', 'IFSC', 'Amount', 'Cheque MICR Number', 'Date', 'Signature']
    print("\nSaving results to Excel...")
    
    # Create a DataFrame with one row
    df = pd.DataFrame([cheque_fields])
    
    # Reorder columns (ensure all exist)
    for col in columns_order:
        if col not in df.columns:
            df[col] = ""
    df = df[columns_order]
    
    output_path = '../Cheque_details.xlsx'
    
    # Use XlsxWriter engine
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    
    worksheet = writer.sheets['Sheet1']
    
    # Adjust column widths
    worksheet.set_column('A:A', 20) # BankName
    worksheet.set_column('B:B', 20) # AC/NO
    worksheet.set_column('C:C', 15) # IFSC
    worksheet.set_column('D:D', 15) # Amount
    worksheet.set_column('E:E', 30) # MICR
    worksheet.set_column('F:F', 15) # Date
    worksheet.set_column('G:G', 20) # Signature
    
    writer.close()
    print(f"Done! Results saved to {output_path}")

if __name__ == '__main__':
    main()
