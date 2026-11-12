# Satellite Image Change Detection using Digital Image Processing (DIP)

A Digital Image Processing semester project developed for detecting land-cover, construction, and urban changes between multi-temporal satellite images using conventional computer vision techniques (without heavy deep-learning dependencies).

---

## 🌟 Key Features

- **Interactive Web Dashboard**:
  - **Split-Curtain Slider**: Drag an interactive divider handle across the viewer to inspect subtle before/after changes.
  - **Synchronized 4-Panel Grid**: Inspect Date 1 (Before), Date 2 (After/Aligned), Difference Heatmap, and Cleaned Binary Change Mask simultaneously.
  - **Alpha-Blended Change Overlay**: Overlay the detected changes directly onto the satellite image with customizable color (Red, Green, Cyan, Yellow, etc.) and transparency slider ($0\% - 100\%$).
  - **Blink / Flicker Comparator**: Toggle between multi-temporal acquisitions at variable speeds (astronomical & remote sensing validation technique).
  - **Real-Time Intensity Histogram**: Interactive Chart.js difference histogram with dynamic Otsu cutoff threshold marker.
  - **Instant Sample Loading**: Single-click demo loading using pre-bundled multi-temporal satellite datasets.
- **Image Registration (ECC)**: Automatically aligns images from different dates using Enhanced Correlation Coefficient (Homography / Affine / Euclidean) to mitigate geometric misalignment and avoid false edge detections.
- **Absolute Image Differencing**: Calculates per-pixel luminance difference and Euclidean RGB intensity maps.
- **Adaptive & Manual Thresholding**: Dynamically separates significant land-cover change from background noise using Otsu's method, Triangle method, or interactive manual slider ($0.01 - 0.99$).
- **Morphological Processing & Blob Filtering**: Applies morphological opening, closing, and connected-component area filtering to eliminate isolated speckle noise and retain coherent change areas.
- **Export Suite**: Export isolated binary change masks, colored overlays, and high-resolution 5-panel composite reports with a single click.
- **Command-Line Interface (CLI)**: Standalone pipeline script for batch processing and headless experimentation.

---

## 📁 Project Structure

```text
Dip_project/
├── app.py                  # Flask web application & REST API endpoints
├── requirements.txt        # Python package dependencies
├── .gitignore              # Files and folders excluded from Git
├── README.md               # Project documentation
├── src/
│   ├── __init__.py
│   ├── utils.py            # Core DIP algorithms (Registration, Differencing, Thresholding, Cleaning, Visuals)
│   └── pipeline.py         # Standalone CLI processing pipeline
├── templates/
│   ├── index.html          # Interactive single-page DIP dashboard
│   └── result.html         # Legacy export view
├── data/                   # Sample satellite imagery (img1.jpg, img2.jpg)
├── uploads/                # Temporary session uploads
├── static/
│   └── results/            # Generated change-detection visual outputs
└── reports/                # CLI visual and evaluation outputs
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Setup Virtual Environment
```powershell
# Clone the repository
git clone <your-repository-url>
cd Dip_project

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
# source .venv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Option A: Streamlit Web Dashboard
Run the Streamlit application:
```powershell
streamlit run streamlit_app.py
```
Open your browser and navigate to:
👉 **[http://localhost:8501](http://localhost:8501)**

Features included in Streamlit:
- Single-click preloaded sample data (Dubai Island) or custom image pair upload.
- ECC Homography registration toggle.
- Thresholding methods: Otsu (Automatic), Triangle (Automatic), or Manual Slider.
- Morphological cleaning (Opening, Closing, Kernel size, and Minimum blob size filtering).
- 4-Panel Analysis Grid, Interactive Comparison (Synchronized / Alpha-Blend Sweep / Overlay sweep).
- Difference intensity histogram with Otsu cutoff reference marker.
- 1-click downloads for binary change masks, colored overlays, and high-res 5-panel composite figures.

---

### Option B: Flask Interactive Web Dashboard
Start the Flask web server:
```powershell
python app.py
```
Open your browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)** (or `http://localhost:5000`)

---

### Option C: Run via Command-Line (CLI)
To run the automated script directly on local image files:
```powershell
python src/pipeline.py data/img1.jpg data/img2.jpg --register
```
Outputs are saved directly to `reports/change_detection.png`.

**Additional Options:**
- `--register`: Enables automatic spatial alignment using homography-based ECC.
- `--thresh_method`: Choose between `otsu` (default) or `fixed`.
- `--thresh_value`: Specify a manual threshold float between `0.0` and `1.0` if using `--thresh_method fixed`.
- `--min_blob`: Minimum connected-component pixel area to retain (default: 30).
- `--out_dir`: Directory where the result plot is stored (default: `reports`).

---

## 🔬 Methodology Overview
```
Satellite Image (Date 1) + Satellite Image (Date 2)
                   │
                   ▼
       Spatial Registration (ECC Homography)
                   │
                   ▼
           Image Differencing (D = |I₂ - I₁|)
                   │
                   ▼
         Thresholding (Otsu / Triangle / Manual)
                   │
                   ▼
     Morphological Processing (Opening & Closing)
                   │
                   ▼
      Connected Component Filtering (Area >= Min)
                   │
                   ▼
   Interactive Visualizer (Split Curtain, 4-Panel Grid, Overlay, Blink)
   & Quantitative Analytics (% Changed, Cluster Count, Intensity Histogram)
```
