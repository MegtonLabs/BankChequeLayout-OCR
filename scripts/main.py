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

import re

def validate_signature(crop):
    """
    Checks if the signature crop actually contains a signature (ink).
    Removes horizontal lines (underlines) before checking.
    Returns True if valid, False otherwise.
    """
    if crop is None or crop.size == 0:
        return False
        
    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Threshold to binary (inverted: text is white, background black)
    # Use simple thresholding to avoid noise from adaptive
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Remove horizontal lines (underline)
    # Increase kernel size to remove longer lines
    # Use a slightly thicker kernel to catch thicker lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2))
    
    # Detect lines
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Subtract lines from binary image
    detected_lines = cv2.dilate(detected_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    binary_no_lines = cv2.bitwise_and(binary, binary, mask=cv2.bitwise_not(detected_lines))
        
    # Check for connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_no_lines, connectivity=8)
    
    valid_components = 0
    total_ink_area = 0
    
    for i in range(1, num_labels): # Skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        width = stats[i, cv2.CC_STAT_WIDTH]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        
        # Filter small noise
        if area > 200:
            # Calculate Aspect Ratio
            aspect_ratio = width / height
            
            # Calculate Density (Solidity within bounding box)
            density = area / (width * height)
            
            # Heuristics to reject non-signature components:
            # 1. Too flat (Line remnant): AR > 5
            # 2. Too solid (Block of text or thick line): Density > 0.6
            
            if aspect_ratio < 5.0 and density < 0.6:
                valid_components += 1
                total_ink_area += area
            
    # Thresholds:
    # At least 1 significant component that isn't a flat line or block
    if valid_components < 1:
        return False
        
    return True

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
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            
            print(f"  - {cls_name} ({conf:.2f})")

            # Draw bounding box on the annotated image (for check_cont_.jpg)
            cv2.rectangle(annotated_img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)

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
                     # Set to True if signature is detected AND valid (has ink)
                     # AND doesn't just say "Authorized Signatory"
                     
                     is_valid_visual = validate_signature(crop)
                     
                     if is_valid_visual:
                         # Run OCR to check for printed labels
                         # Preprocess? Maybe not needed for printed text, but let's be safe
                         # processed_sig = preprocess_image(crop, 'Signature') # We don't have a specific one, default is fine
                         
                         print(f"    Validating Signature content with OCR...")
                         sig_text_list = vision_api(crop)
                         sig_text = " ".join(sig_text_list).strip().upper()
                         print(f"    Signature OCR: {sig_text}")
                         
                         # Keywords that indicate it's just a label
                         keywords = [
                             "SIGNATORY", "AUTHORIZED", "AUTHORISED", "MANAGER", "DIRECTOR", 
                             "SECRETARY", "CHAIRMAN", "PRESIDENT", "TREASURER", "PARTNER", 
                             "TRUSTEE", "PROPRIETOR", "AUTH", "SIGN", "HERE", "ABOVE", "BELOW",
                             "PARTNERSHIP", "HOLDER", "SHIP", "FOR"
                         ]
                         
                         is_label = False
                         for kw in keywords:
                             if kw in sig_text:
                                 is_label = True
                                 break
                                 
                         if is_label:
                             print(f"    Signature rejected (Label detected: {sig_text})")
                             cheque_fields['Signature'] = 'False'
                         else:
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
    cv2.imwrite('../fields/check_cont_.jpg', annotated_img)

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
