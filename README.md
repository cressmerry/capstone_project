# InsightSSD Object Detection System

InsightSSD is an interactive object detection application that combines a pre-trained **Single Shot MultiBox Detector (SSD)** with a lightweight frontend dashboard. It features an OpenCV DNN backend for inference, customizable filtering thresholds, class category selection, and support for both uploaded images and live camera streams.

The repository also includes standalone pipelines and Jupyter notebooks for custom training/fine-tuning models on everyday object categories from the Google Open Images dataset.

---

## Project Structure

```text
├── backend/                   # FastAPI Server & OpenCV Inference Engine
│   ├── models/                # Downloaded weights directory (git-ignored)
│   ├── app.py                 # FastAPI endpoints & static routing
│   ├── model.py               # OpenCV DNN SSD wrapper class
│   ├── download_models.py     # Script to pull weights/configs from TensorFlow/GitHub
│   └── requirements.txt       # Python backend dependencies
├── frontend/                  # React & Vite Dashboard Dashboard
│   ├── src/                   # React components & app entry points
│   └── package.json           # Frontend scripts and dependencies
├── src/                       # Custom SSD training utilities
│   ├── dataset.py             # Open Images dataset downloader & formatter
│   └── train.py               # Custom TensorFlow training loop script
├── notebook/                  # Jupyter Lab Walkthroughs
│   └── train_ssd_openimages.ipynb  # Interactive fine-tuning notebook
└── README.md                  # Project-wide documentation (this file)
```

---

## System Requirements

- **Python**: version 3.8 or higher.
- **Node.js**: version 18 or higher.
- **npm**: version 9 or higher.
- **Web Camera**: (Optional) For real-time inference features.

---

## 1. Backend Setup & Launch

The backend is built with **FastAPI** and uses **OpenCV (DNN module)** for running inference.

### Step 1.1: Create a Python Virtual Environment
Navigate to the project root directory and create/activate a virtual environment:

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate
```

### Step 1.2: Install Dependencies
Install the required packages listed in `backend/requirements.txt`:

```bash
pip install -r backend/requirements.txt
```

### Step 1.3: Download Pre-trained Weights & Labels
Before launching the server, you must download the SSD MobileNet V2 weights and label configuration. A utility script is provided to automate this:

```bash
python backend/download_models.py
```
This script downloads:
1. `frozen_inference_graph.pb` (SSD MobileNet V2 weights trained on COCO dataset)
2. `ssd_mobilenet_v2_coco_2018_03_29.pbtxt` (OpenCV text configuration file)
3. Saves a local labels dictionary `coco_labels.json`.

### Step 1.4: Launch the FastAPI server
Start the backend API using Uvicorn:

```bash
# Run from root directory
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```
- **API Status URL**: [http://127.0.0.1:8000/api/status](http://127.0.0.1:8000/api/status)
- **Interactive API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 2. Frontend Setup & Run (Development)

The frontend is built with **React**, **Vite**, and styled with custom CSS glassmorphism components.

### Step 2.1: Navigate to Frontend Directory & Install Modules
Open a new terminal window, ensure Node.js is installed, and run:

```bash
cd frontend
npm install
```

### Step 2.2: Launch Vite Development Server
Run the local dev server:

```bash
npm run dev
```
By default, the dashboard runs at [http://localhost:5173/](http://localhost:5173/). The application will connect directly to the FastAPI server running on port 8000.

---

## 3. Production Deployment (Single-Server Mode)

For deployment, you can build the frontend static assets and serve them directly from the FastAPI server instead of running two separate dev processes.

### Step 3.1: Build the React Application
In the `frontend` directory, build the distribution bundle:

```bash
cd frontend
npm run build
```
This compilation creates a `frontend/dist/` directory containing all optimized HTML, JS, and CSS files.

### Step 3.2: Run FastAPI
The FastAPI app mounts `frontend/dist/` at the root path `/`. Simply launch the backend server:

```bash
# Run from root directory
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser to view the fully deployed, self-contained dashboard.

---

## 4. Custom Model Training & Exploration

To train or fine-tune models from scratch on the Open Images dataset:

### Jupyter Notebook Walkthrough
Launch Jupyter Lab or Notebook to run the fine-tuning tutorial interactively:
```bash
jupyter lab
```
Navigate to `notebook/train_ssd_openimages.ipynb`. It details:
- Downloading subsets of the Open Images dataset.
- Parsing annotations.
- Implementing a MultiBox Loss function.
- Fine-tuning the SSD detection head.

### CLI Pipeline Scripts
- **Download Dataset**: Use `src/dataset.py` to target specific everyday object categories (such as person, laptop, chair, book) and download images/metadata from Google Cloud Storage:
  ```bash
  python src/dataset.py
  ```
- **Train SSD**: Run the TensorFlow training loop to train the Single Shot MultiBox Detector head:
  ```bash
  python src/train.py
  ```
