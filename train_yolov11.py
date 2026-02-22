"""
YOLOv11 Cheque Detection Training Script
This script trains a YOLOv11 model on the Cheque dataset.
"""

import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import colors  # Imported the plotting utility
import os
import yaml


def main():
    # Configuration
    dataset_path = 'DataSet'
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    model_size = 's'  # Options: n, s, m, l, x (nano, small, medium, large, xlarge)
    
    # Print system info
    print('='*50)
    print('YOLOv11 Cheque Detection Training')
    print('='*50)
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print('='*50)
    
    # Verify dataset
    if not os.path.exists(data_yaml_path):
        print(f'ERROR: data.yaml not found at {data_yaml_path}')
        return
    
    # Read dataset config
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
        num_classes = data_config['nc']
        class_names = data_config['names']
        # Read custom colors from data.yaml if available
        custom_colors = data_config.get('colors', None)
    
    # --- Custom Color Override for Plotting ---
    custom_palette = [
        (0, 0, 255),      # 0: Account_Number - Blue
        (255, 0, 0),      # 1: Amount - Red
        (255, 140, 0),    # 2: Bank_Name - Orange
        (255, 0, 255),    # 3: Date - Magenta
        (255, 255, 0),    # 4: IFSC - Yellow
        (128, 0, 128),    # 5: MICR - Purple
        (0, 128, 0)       # 6: Signature - Dark Green
    ]
    colors.palette = custom_palette
    colors.n = len(custom_palette)
    # ------------------------------------------

    print(f'\nDataset Configuration:')
    print(f'  Number of classes: {num_classes}')
    print(f'  Class names: {class_names}')
    if custom_colors:
        print(f'  ✓ Custom colors loaded from data.yaml')
    print('='*50)
    
    # Load YOLOv11 model
    print(f'\nLoading YOLOv11{model_size} model...')
    model = YOLO(f'yolo11{model_size}.pt')
    print('Model loaded successfully!')
    print('\nNOTE: Custom colors from data.yaml will be used for all training visualizations')
    
    # Train the model
    print('\nStarting training...')
    print('='*50)
    
    model.train(
        data=data_yaml_path,
        epochs=200,
        imgsz=416,
        batch=16,
        # name='weights',
        patience=50,
        save=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=8,
        # project='runs/detect',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        seed=0,
        deterministic=True,
        amp=True,
        # Learning rate settings
        lr0=0.005,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # Loss gains
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=10
    )
    
    print('\n' + '='*50)
    print('Training completed!')
    print('='*50)
    
    # Validate the model
    print('\nRunning validation...')
    metrics = model.val()
    
    print(f'\nValidation Results:')
    print(f'  mAP50: {metrics.box.map50:.4f}')
    print(f'  mAP50-95: {metrics.box.map:.4f}')
    print(f'  Precision: {metrics.box.mp:.4f}')
    print(f'  Recall: {metrics.box.mr:.4f}')
    print('='*50)
    
    print(f'\nModel saved to: runs/detect/cheque_detection/weights/best.pt')
    print('\nTraining complete! Check runs/detect/cheque_detection/ for results.')


if __name__ == '__main__':
    main()