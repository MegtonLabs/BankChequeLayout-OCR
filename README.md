# Bank Cheque Layout OCR

A comprehensive Computer Vision system designed to automate the extraction of key information from bank cheques. This project leverages **YOLOv11** for precise layout detection and **GLM-OCR via Ollama** for intelligent text extraction.

## 🚀 Features

* **Object Detection:** Accurate detection of layout fields (Date, Account Number, Amount, Signature, Bank Name, MICR, IFSC) using **YOLOv11**.
* **OCR Analysis:** Advanced text extraction using **GLM-OCR** (running through Ollama).
* **Automated Reporting:** Generates detailed Excel reports (`Cheque_details.xlsx`) with extracted data.
* **Visual Output:** Saves annotated images with color-coded bounding boxes for verification.
* **Field Preprocessing:** Intelligent image preprocessing optimized for each field type.
* **Signature Validation:** Detects and validates actual signatures vs. printed labels.

## 🛠️ Installation & Setup

### Prerequisites

Before you begin, ensure you have:
- **Ollama** installed - [Download](https://ollama.ai/)
- **Anaconda/Miniconda** - [Download](https://docs.conda.io/en/latest/miniconda.html)
- **Python 3.11** recommended

### Step 1: Set up Ollama with GLM-OCR

The project uses **GLM-OCR via Ollama** for text extraction. Install and configure it first:

```bash
# 1. Install Ollama (download from https://ollama.ai/ and install)

# 2. Download the GLM-OCR model (run in terminal/cmd):
ollama pull glm-ocr

# 3. Start the Ollama server (keep this running in a separate terminal):
ollama serve

# Ollama will be available at: http://localhost:11434
```

⚠️ **Important:** Keep the Ollama server running while using the cheque OCR system.

### Step 2: Create and Activate Conda Environment

```bash
# 1. Create a new conda environment
conda create -n cheque-ocr python=3.11 -y

# 2. Activate the environment
conda activate cheque-ocr

# 3. Navigate to project directory
cd BankChequeLayout-OCR

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install YOLOv11 (Ultralytics)
pip install ultralytics
```

### Step 3: Verify Setup

```bash
# Test Ollama connection
python -c "import requests; print('✓ Ollama OK' if requests.get('http://localhost:11434/api/tags').status_code == 200 else '✗ Ollama Not Running')"
```

---

## ⚙️ Configuration

### GLM-OCR Settings

The project uses **Ollama GLM-OCR** for OCR tasks. Configuration is in `scripts/vision.py`:

```python
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Default Ollama endpoint
OLLAMA_MODEL = "glm-ocr"  # Model name
```

**To change the Ollama endpoint:**
- Edit `OLLAMA_API_URL` in `scripts/vision.py` if running Ollama on a different machine
- Example: `"http://192.168.1.100:11434/api/generate"`

### YOLOv11 Model

The project uses a trained **YOLOv11s** model (`weights/best.pt`) for field detection. No configuration needed for inference.

---

## 🏋️ Training the Model

To retrain the YOLOv11 model on new cheque data:

### Prerequisites
- Dataset organized in `DataSet/` directory
- `DataSet/data.yaml` configured with dataset paths
- Training data split into train/valid/test folders

### Training Command

```bash
# Run from project root directory
python train_yolov11.py
```

**Training Configuration:**
- Model: YOLOv11s (Small)
- Epochs: 200
- Batch Size: 16
- Image Size: 416x416
- Device: GPU (if available) or CPU
- Output: `runs/detect/cheque_detection/weights/best.pt`

**Features:**
- Custom color visualization for each field type
- Early stopping (patience: 50 epochs)
- Data augmentation enabled
- Automatic checkpoint saving

---

## ⚡ Running Inference (Usage)

### Option 1: Batch Processing (All Test Images)

Process all images in the test dataset and generate a CSV report:

```bash
# Run from project root
python inference_yolov11.py
```

**Output:**
- Annotated images: `runs/detect/predictions/`
- Detection results: `runs/detect/detection_results.csv`

### Option 2: Single Image Processing (With OCR)

Extract text and generate detailed Excel report for a single cheque:

```bash
# Navigate to scripts directory first
cd scripts

# Run on a single image
python main.py --input_image "../cheques/sample_cheque.jpg"
```

**Output:**
- Annotated image: `fields/check_cont_.jpg`
- Extracted fields: `Cheque_details.xlsx` (in root directory)
- Individual field crops: `fields/` directory

**Example:**

```bash
python main.py --input_image "../cheques/Cheque_6.jpg"
```

---

## 📂 Project Structure

```
BankChequeLayout-OCR/
├── DataSet/                          # Training and test datasets
│   ├── train/                        # Training images and labels
│   ├── valid/                        # Validation images and labels
│   ├── test/                         # Test images
│   └── data.yaml                     # Dataset configuration
├── scripts/                          # Core Python scripts
│   ├── main.py                       # Single image processing pipeline
│   ├── vision.py                     # GLM-OCR integration (via Ollama)
│   ├── preprocess.py                 # Image preprocessing
│   ├── extract_*.py                  # Field-specific extraction scripts
│   ├── micr_ocr.py                   # MICR recognition
│   └── imports.py                    # Common imports
├── weights/                          # Trained YOLOv11 model weights
│   ├── best.pt                       # Best trained model
│   └── last.pt                       # Last checkpoint
├── runs/                             # Training and inference outputs
│   └── detect/
│       ├── cheque_detection/         # Training logs
│       └── predictions/              # Inference results
├── fields/                           # Extracted field crops (temporary)
├── cheques/                          # Sample cheque images
├── train_yolov11.py                  # Training script
├── inference_yolov11.py              # Batch inference script
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── MIGRATION_TO_GLMOCR.md            # Migration guide from TrOCR
```

---

## 🔍 Extracted Fields

The system extracts and processes the following cheque fields:

| Field | Detection | OCR | Output |
|-------|-----------|-----|--------|
| **Bank Name** | ✅ YOLOv11 | ✅ GLM-OCR | Excel Column |
| **Account Number** | ✅ YOLOv11 | ✅ GLM-OCR | Excel Column |
| **IFSC Code** | ✅ YOLOv11 | ✅ GLM-OCR + Validation | Excel Column |
| **Amount** | ✅ YOLOv11 | ✅ GLM-OCR + Formatting | Excel Column (₹ format) |
| **Date** | ✅ YOLOv11 | ✅ GLM-OCR + Parsing | Excel Column (DD/MM/YYYY) |
| **MICR Number** | ✅ YOLOv11 | ✅ Template Matching | Excel Column |
| **Signature** | ✅ YOLOv11 | ✅ Visual Validation + OCR | Boolean (True/False) |

---

## 📊 Performance Metrics

**YOLOv11s Model:**
- Average Precision (mAP50): ~0.90+
- Detection Speed: ~50ms per image (GPU)
- Supported Input Sizes: 416x416 (optimized)

**GLM-OCR (via Ollama):**
- Text Recognition Accuracy: Excellent on printed and handwritten text
- Processing Speed: ~100-200ms per field (depends on Ollama hardware)
- Supports: Printed text, handwritten digits, special characters

---

## ⚠️ Troubleshooting

### Ollama Connection Error
```
ERROR: Cannot connect to Ollama. Make sure Ollama is running...
```
**Solution:**
```bash
# Terminal 1: Start Ollama server
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

### GLM-OCR Model Not Found
```
Error: Model 'glm-ocr' not found
```
**Solution:**
```bash
ollama pull glm-ocr
```

### Poor OCR Results
- Ensure input image has good contrast
- Check that Ollama server is not overloaded
- Try preprocessing parameters in `main.py` (CLAHE settings)

### Model Not Loading
```bash
# Ensure YOLO model exists
ls weights/best.pt

# If missing, download or retrain:
python train_yolov11.py
```

---

## 🚀 Quick Start Guide

```bash
# 1. Install Ollama and pull GLM-OCR
ollama pull glm-ocr

# 2. Start Ollama server (keep in separate terminal)
ollama serve

# 3. Create and activate environment
conda create -n cheque-ocr python=3.11 -y
conda activate cheque-ocr

# 4. Install dependencies
pip install -r requirements.txt
pip install ultralytics

# 5. Process a cheque
cd scripts
python main.py --input_image "../cheques/sample_cheque.jpg"

# 6. Check results
# Output: ../Cheque_details.xlsx
```
