# InsightSSD Workplace Detector

InsightSSD Workplace Detector is an interactive object detection dashboard and deep learning laboratory designed for professional office automation, workplace safety, and ergonomics monitoring. It features a high-performance Python backend server combined with a premium React dark-mode dashboard.

The application allows users to perform real-time inference on uploaded office photos or live camera streams, with dynamic model selection between a general-purpose **COCO Pre-trained Model** (80 classes) and a custom **Fine-Tuned Workplace Model** (9 classes).

---

## Project Structure

```text
├── backend/                   # FastAPI Server & OpenCV Inference Engine
│   ├── models/                # Downloaded COCO weights directory (git-ignored)
│   ├── app.py                 # FastAPI endpoints & static assets routing
│   ├── model.py               # Dual-inference (OpenCV DNN & TensorFlow) model wrapper
│   ├── download_models.py     # Script to pull base pre-trained COCO models
│   └── requirements.txt       # Python backend dependencies
├── dataset/                   # Local dataset directory (git-ignored)
│   ├── images/                # Downloaded raw images (.jpg format)
│   ├── annotations.json       # Formatted validation bounding boxes
│   └── ssd_fine_tuned_head.weights.h5 # Fine-tuned custom model weights
├── frontend/                  # React & Vite Dashboard Frontend
│   ├── src/                   # React components (Header, Parameters, Tabs, etc.)
│   └── package.json           # Frontend packages and build scripts
├── src/                       # Custom SSD training pipeline
│   ├── dataset.py             # Parallel Open Images dataset downloader & parser
│   └── train.py               # tf.data.Dataset based transfer learning pipeline
├── notebook/                  # Jupyter Lab walkthroughs
│   └── train_ssd_openimages.ipynb  # Interactive fine-tuning notebook
└── README.md                  # Project-wide documentation (this file)
```

---

## System Requirements

- **Python**: Version 3.9 to 3.12 (Python 3.13 is NOT supported by TensorFlow).
- **Node.js**: Version 18 or higher.
- **npm**: Version 9 or higher.
- **Web Camera**: (Optional) For real-time workspace monitoring.

---

## 1. Setup & Installation

### Step 1.1: Backend Virtual Environment
Navigate to the project root directory and set up a virtual environment containing Python dependencies. **IMPORTANT: You must use Python 3.9 - 3.12. Creating the venv with python3.13+ will fail during dependency installation.**

**On Linux/macOS:**
```bash
# Create the virtual environment (Must use Python 3.9 - 3.12)
python3.10 -m venv .venv  # Or python3.9, python3.11, python3.12

# Activate the virtual environment
source .venv/bin/activate

# Install required packages
pip install -r backend/requirements.txt
```

**On Windows:**
```cmd
:: Create the virtual environment (Must use Python 3.9 - 3.12)
py -3.11 -m venv .venv  :: Or -3.9, -3.10, -3.12

:: Activate the virtual environment (Command Prompt)
.venv\Scripts\activate.bat

:: Activate the virtual environment (PowerShell)
.venv\Scripts\Activate.ps1

:: Install required packages
pip install -r backend/requirements.txt
```

### Step 1.2: Download Base COCO Models
Before launching the server, you need to download the base SSD MobileNet V2 weights and configs:

```bash
python backend/download_models.py
```
This utility script downloads:
1. `frozen_inference_graph.pb` (Pre-trained COCO weights)
2. `ssd_mobilenet_v2_coco_2018_03_29.pbtxt` (OpenCV DNN configurations)
3. Saves a local labels mapping file `coco_labels.json`.

### Step 1.3: Frontend Dependencies
Navigate to the frontend folder and install Node.js modules:

```bash
cd frontend
npm install
```

---

## 2. Workplace Fine-Tuning Pipeline

The pipeline is built to train a specialized detector head for everyday office-based automation. The 9 targeted target classes are:
- `person`, `laptop`, `chair`, `book`, `dining table` (mapped from Open Images `table`), `cell phone`, `cup`, `keyboard`, `mouse`.

### Step 2.1: Exhaustive Data Acquisition
To compile the training dataset, run the downloader script:
```bash
python src/dataset.py
```
* **How it works**: The script downloads the official Google Open Images validation split bounding boxes and metadata, filters them for the 9 targeted office classes, and retrieves images in parallel using `concurrent.futures.ThreadPoolExecutor` (20 worker threads).
* **Console Progress**: Uses `tqdm` to display download speeds, ETA, and progress bars. Bounding box coordinates are saved in normalized `[ymin, xmin, ymax, xmax]` format inside `dataset/annotations.json`.

### Step 2.2: TensorFlow Model Training
Run the training script to train the custom SSD head:
```bash
python src/train.py
```
* **tf.data.Dataset Input Pipeline**: Refactored to adhere to industry-standard deep learning practices. Instead of loading full images into RAM, it streams paths, decodes JPEGs natively, resizes them, and normalizes them in parallel on CPU worker threads using `tf.data.AUTOTUNE` and `.prefetch()`.
* **Architecture**: Freezes the pre-trained `MobileNetV2` feature extractor backbone and appends two custom output heads:
  1. **Classification Head**: Dense layer outputting probability distribution over 9 target classes.
  2. **Bounding Box Head**: Regression layer outputting normalized bounding box coordinates.
* **Loss Function**: Custom MultiBox Loss combining Categorical Crossentropy and Smooth L1 Loss.
* **Result**: Saves the output weights to `dataset/ssd_fine_tuned_head.weights.h5`.

---

## 3. Running the Application

### Option A: Local Development Mode (Separate Servers)
Run both components concurrently in separate terminals for fast hot-reloading:

* **Start Backend**:
  * **On Linux/macOS**:
    ```bash
    source .venv/bin/activate
    uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
    ```
  * **On Windows**:
    ```cmd
    .venv\Scripts\activate
    uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
    ```
* **Start Frontend**:
  ```bash
  cd frontend
  npm run dev
  ```
  Open [http://localhost:5173/](http://localhost:5173/) in your web browser.

### Option B: Production Deployment Mode (Single Server)
FastAPI can serve the React frontend static pages directly:

1. **Build Frontend**:
   ```bash
   cd frontend
   npm run build
   ```
2. **Launch Integrated Server**:
   ```bash
   # From root directory
   uvicorn backend.app:app --host 127.0.0.1 --port 8000
   ```
   Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser. The dashboard automatically syncs with the FastAPI detection endpoints.

---

## 4. Interactive Testing
- **Model Switching**: Toggle between "COCO" and "Fine-Tuned SSD" on the dashboard settings. The backend automatically loads the fine-tuned `.weights.h5` head when the Custom model is requested.
- **Interactive Checklists**: Use the scrollable checklist panel to selectively isolate and display targets (e.g. tracking computer peripheral placement or human presence).
