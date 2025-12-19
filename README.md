# Bank Cheque Layout OCR

A comprehensive Computer Vision system designed to automate the extraction of key information from bank cheques. This project leverages **YOLOv11** for precise layout detection and integrates OCR technologies to digitize extracted fields.

![YOLOv11 Architecture](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo-comparison-plots.png)

## 🚀 Features
*   **Object Detection:** Accurate detection of layout fields (Date, Account Number, Amount, Signature, Bank Name, MICR, IFSC) using **YOLOv11**.
*   **OCR Analysis:** Text extraction using Tesseract and other OCR utilities.
*   **Automated Reporting:** Generates detailed Excel reports (`Cheque_details.xlsx`) with extracted data.
*   **Visual Output:** Saves annotated images with bounding boxes for verification.

## 🛠️ Installation & Setup

We recommend using **Conda** to manage the dependencies and environment.

### 1. Prerequisites
Ensure you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

### 2. Create Conda Environment

Open your terminal or Anaconda Prompt and run the following commands to set up the environment:

```bash
# 1. Create a new conda environment (Python 3.10 recommended)
conda create -n cheque-ocr python=3.11 -y

# 2. Activate the environment
conda activate cheque-ocr

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install YOLOv11 (Ultralytics)
pip install ultralytics

# (Optional) For GPU support (Recommended for training):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🏋️ Training the Model

If you want to retrain the model on new data:

1.  Ensure your dataset is prepared in `DataSet/` and configured in `DataSet/data.yaml`.
2.  Run the training script from the **root directory**:

```bash
python train_yolov11.py
```
*   **Model:** YOLOv11s (Small)
*   **Epochs:** 100
*   **Optimizer:** SGD
*   **Output:** Best weights saved to `runs/detect/cheque_detection/weights/best.pt`

---

## ⚡ Running Inference (Usage)

### 1. Batch Processing (Test Dataset)
To run object detection on all images in `DataSet/test/images` and generate a CSV report:

```bash
python inference_yolov11.py
```
*Results will be saved in `runs/detect/predictions` and `runs/detect/detection_results.csv`.*

### 2. Single Image Check (Interface)
To process a specific cheque image, extract text (OCR), and generate a report, use the main pipeline script.

**Important:** Navigate to the `scripts` directory first.

```bash
cd scripts
python main.py --input_image "../path/to/your/cheque_image.jpg"
```

**Example:**
```bash
python main.py --input_image "../cheques/sample_cheque.jpg"
```

*   **Output:**
    *   Annotated Image: `fields/check_cont_.jpg`
    *   Excel Report: `Cheque_details.xlsx` (in the root directory)

---

## 📂 Project Structure
*   `DataSet/`: Training and Testing data.
*   `runs/`: Model training logs and inference outputs.
*   `scripts/`: Core analysis scripts (`main.py`, `vision.py`, `preprocessing.py`).
*   `weights/`: Contains trained model weights (`yolo11s.pt`).
*   `inference_yolov11.py`: Batch inference script.
*   `train_yolov11.py`: Training entry point.

---

## 📊 Performance
The model utilizes **YOLOv11s** and achieves high detection accuracy across standard cheque fields.
*   **Optimization:** SGD with Momentum (0.937)
*   **Loss Function:** Composite (CIoU + BCE + DFL)
*   **Accuracy:** High precision on MICR, Account No, and Date fields.
