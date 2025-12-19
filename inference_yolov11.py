"""
YOLOv11 Cheque Detection Inference Script
This script runs inference on images using the trained YOLOv11 model and saves the detection results to a CSV file.
"""

import torch
from ultralytics import YOLO
import os
import glob
import cv2
from pathlib import Path
import pandas as pd


def main():
    # Configuration
    model_path = 'runs/detect/cheque_detection/weights/best.pt'
    test_images_dir = 'DataSet/test/images'
    output_dir = 'runs/detect/predictions'
    results_csv_path = 'runs/detect/detection_results.csv'
    confidence_threshold = 0.25
    
    print('='*50)
    print('YOLOv11 Cheque Detection Inference')
    print('='*50)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f'ERROR: Model not found at {model_path}')
        print('Please train the model first using train_yolov11.py')
        return
    
    # Load the trained model
    print(f'\nLoading model from: {model_path}')
    model = YOLO(model_path)
    print('Model loaded successfully!')
    
    # Get test images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    test_images = []
    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(test_images_dir, ext)))
    
    if not test_images:
        print(f'\nERROR: No images found in {test_images_dir}')
        return
    
    print(f'\nFound {len(test_images)} test images')
    print(f'Confidence threshold: {confidence_threshold}')
    print(f'Output directory: {output_dir}')
    print('='*50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # List to store detection results
    detection_data = []
    
    # Run inference on each image
    print('\nRunning inference...\n')
    
    for i, image_path in enumerate(test_images, 1):
        image_name = os.path.basename(image_path)
        print(f'[{i}/{len(test_images)}] Processing: {image_name}')
        
        # Run prediction (don't save automatically)
        results = model.predict(
            source=image_path,
            conf=confidence_threshold,
            save=False  # We'll save manually with custom settings
        )
        
        # Plot with custom font size and save
        result_img = results[0].plot(
            line_width=2,     # Box line thickness
            font_size=5.0,    # Larger font (try 1.0-3.0)
            labels=True,
            conf=True
        )
        
        # Save the image
        save_path = os.path.join(output_dir, image_name)
        cv2.imwrite(save_path, result_img)
        
        # Extract and store detection results
        detections = results[0].boxes
        if len(detections) > 0:
            print(f'  Detected {len(detections)} objects:')
            for box in detections:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                xyxy = box.xyxy[0].cpu().numpy()
                detection_data.append({
                    'image_name': image_name,
                    'class_name': cls_name,
                    'confidence': conf,
                    'x1': xyxy[0],
                    'y1': xyxy[1],
                    'x2': xyxy[2],
                    'y2': xyxy[3]
                })
                print(f'    - {cls_name}: {conf:.2f}')
        else:
            print('  No objects detected')
    
    # Save detection data to CSV
    if detection_data:
        print('\n' + '='*50)
        print('Saving detection results to CSV...')
        df = pd.DataFrame(detection_data)
        df.to_csv(results_csv_path, index=False)
        print(f'Results saved to: {results_csv_path}')
    else:
        print('\nNo detections to save.')
        
    print('\n' + '='*50)
    print('Inference completed!')
    print(f'Annotated images saved to: {output_dir}')
    print('='*50)


if __name__ == '__main__':
    main()
