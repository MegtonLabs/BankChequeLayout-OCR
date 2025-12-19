import cv2
import numpy as np

def analyze(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Image not found: {path}")
        return

    # Invert so text/lines are white (255)
    # The debug image is already binary (black text on white bg usually, or wait)
    # adaptiveThreshold returns 255 for pixels > threshold.
    # If text is black on white, then text is 0.
    # So we need to invert to count "ink".
    img_inv = cv2.bitwise_not(img)
    
    # Vertical projection (sum of rows? No, sum of columns)
    # Axis 0 is rows (y), Axis 1 is cols (x).
    # Sum along axis 0 to get column sums.
    proj = np.sum(img_inv, axis=0) / 255.0 # Count of white pixels in each column
    
    height = img.shape[0]
    
    # Find peaks that are close to full height
    # e.g. > 80% of height
    threshold = height * 0.8
    peaks = np.where(proj > threshold)[0]
    
    print(f"File: {path}")
    print(f"  Height: {height}")
    print(f"  Max Peak: {np.max(proj)}")
    print(f"  Num Peaks (>80%): {len(peaks)}")
    print(f"  Peak Locations: {peaks}")
    print("-" * 20)

analyze('../fields/debug_date_img.jpg')
analyze('../fields/debug_date_img2.jpg')
