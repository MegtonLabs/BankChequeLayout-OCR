import cv2
import numpy as np

img = cv2.imread('../fields/debug_signature_binary.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Debug image not found.")
    exit()

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

print(f"Total components: {num_labels}")
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    width = stats[i, cv2.CC_STAT_WIDTH]
    height = stats[i, cv2.CC_STAT_HEIGHT]
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    
    print(f"Component {i}: Area={area}, W={width}, H={height}, Pos=({x},{y})")
