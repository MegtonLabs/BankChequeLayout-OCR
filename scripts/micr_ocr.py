from imports import *
import cv2
import numpy as np
import imutils
from imutils import contours

def extract_digits_and_symbols(image, charCnts, minW=5, minH=15):
    charIter = charCnts.__iter__()
    rois = []
    locs = []

    while True:
        try:
            c = next(charIter)
            (cX, cY, cW, cH) = cv2.boundingRect(c)
            roi = None

            if cW >= minW and cH >= minH:
                roi = image[cY:cY + cH, cX:cX + cW]
                rois.append(roi)
                locs.append((cX, cY, cX + cW, cY + cH))
            else:
                parts = [c, next(charIter), next(charIter)]
                (sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf, -np.inf)

                for p in parts:
                    (pX, pY, pW, pH) = cv2.boundingRect(p)
                    sXA = min(sXA, pX)
                    sYA = min(sYA, pY)
                    sXB = max(sXB, pX + pW)
                    sYB = max(sYB, pY + pH)

                roi = image[sYA:sYB, sXA:sXB]
                rois.append(roi)
                locs.append((sXA, sYA, sXB, sYB))

        except StopIteration:
            break

    return (rois, locs)

class MICR_OCR:
    def __init__(self, reference_path='../reference_micr.png'):
        self.chars = {}
        self.load_reference(reference_path)
        self.rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))

    def load_reference(self, path):
        if not os.path.exists(path):
            print(f"Warning: Reference MICR image not found at {path}")
            return

        # reference characters
        # User provided: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "T", "U", "A", "D"]
        # Standard E-13B: 0-9, T (Transit), U (On-Us), A (Amount), D (Dash)
        charNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "T", "U", "A", "D"]

        ref = cv2.imread(path)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        ref = imutils.resize(ref, width=400)
        ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refCnts = imutils.grab_contours(refCnts)
        refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

        # Check if we found the right number of contours
        # print(f"Found {len(refCnts)} reference contours")
        
        refROIs = extract_digits_and_symbols(ref, refCnts, minW=10, minH=20)[0]
        
        if len(refROIs) != len(charNames):
             print(f"Warning: Found {len(refROIs)} chars in reference, expected {len(charNames)}")
             # Fallback or strict mapping? Let's try to map what we have or just proceed.
             # If mismatch, the mapping will be wrong.
             # Assuming the reference image is the standard one used in tutorials.

        for (name, roi) in zip(charNames, refROIs):
            roi = cv2.resize(roi, (36, 36))
            self.chars[name] = roi

    def predict(self, crop_img):
        if not self.chars:
            return ""

        # Resize crop to have a fixed width to ensure consistency with reference
        # The reference chars were extracted from a width=400 image.
        # If we resize the crop to width=800 (since it's the full line), the chars might be too big?
        # User's code resized the *reference* image to 400.
        # And applied it to the *bottom 20%* of the input image (which was presumably large).
        # Let's try resizing the crop to a width that makes characters roughly the same size as reference.
        # If the MICR line is the full width of the cheque, and we resize cheque to X, 
        # then MICR line width is ~X.
        # Let's try resizing to width 600 or 800 for better resolution, 
        # but we need to ensure the contour detection works.
        
        crop_img = imutils.resize(crop_img, width=800)

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to find groups of text
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.rectKernel)
        # User requested 'dilated.jpg', blackhat is close to that step in this context
        cv2.imwrite('../fields/dilated.jpg', blackhat)

        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
        gradX = gradX.astype("uint8")

        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, self.rectKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imwrite('../fields/img_th.jpg', thresh)
        
        # Save a copy as final_mask.jpg as requested
        cv2.imwrite('../fields/final_mask.jpg', thresh)
        
        # Disable clear_border as it might remove valid chars if crop is tight
        # thresh = clear_border(thresh)

        groupCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        groupCnts = imutils.grab_contours(groupCnts)
        groupLocs = []

        for (i, c) in enumerate(groupCnts):
            (x, y, w, h) = cv2.boundingRect(c)
            # Filter small noise
            if w > 10 and h > 10: 
                groupLocs.append((x, y, w, h))

        groupLocs = sorted(groupLocs, key=lambda x: x[0])

        output = []

        for (gX, gY, gW, gH) in groupLocs:
            groupOutput = []
            # Extract the group ROI with some padding
            # Be careful with boundaries
            y1 = max(0, gY - 5)
            y2 = min(gray.shape[0], gY + gH + 5)
            x1 = max(0, gX - 5)
            x2 = min(gray.shape[1], gX + gW + 5)
            
            group = gray[y1:y2, x1:x2]
            group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # Save the processed group (ROI) as final_templ.jpg
            cv2.imwrite(f'../fields/final_templ_{i}.jpg', group)
            cv2.imwrite('../fields/final_templ.jpg', group) # Save the last one as the requested filename

            charCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            charCnts = imutils.grab_contours(charCnts)
            
            if not charCnts:
                continue
                
            charCnts = contours.sort_contours(charCnts, method="left-to-right")[0]

            (rois, locs) = extract_digits_and_symbols(group, charCnts, minW=5, minH=15)

            for roi in rois:
                scores = []
                roi = cv2.resize(roi, (36, 36))

                for charName in self.chars:
                    result = cv2.matchTemplate(roi, self.chars[charName], cv2.TM_CCOEFF)
                    (_, score, _, _) = cv2.minMaxLoc(result)
                    scores.append(score)

                groupOutput.append(list(self.chars.keys())[np.argmax(scores)])

            output.append("".join(groupOutput))

        return " ".join(output)
