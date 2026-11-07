# Satellite Image Change Detection using Digital Image Processing (DIP)

A Digital Image Processing semester project developed for detecting land-cover, construction, and urban changes between multi-temporal satellite images using conventional computer vision techniques (without heavy deep-learning dependencies).

---

## 🌟 Key Features
- **Image Registration (ECC)**: Automatically aligns images from different dates to mitigate geometric misalignment and avoid false boundary detections.
- **Absolute Image Differencing**: Calculates per-pixel luminance difference intensity maps.
- **Adaptive Thresholding (Otsu's Method)**: Dynamically separates significant land-cover change from background noise and minor illumination variations.
- **Morphological Processing**: Applies morphological opening and closing plus connected-component filtering to clean isolated noisy pixels and retain coherent change areas.
- **Interactive Web Interface**: Built with Flask for uploading paired multi-temporal satellite images and viewing side-by-side results.
- **Command-Line Interface (CLI)**: For batch processing and headless experimentation.

---

## 📁 Project Structure

```text
Dip_project/
├── app.py                  # Flask web application entry point
├── requirements.txt        # Python package dependencies
├── .gitignore              # Files and folders excluded from Git
├── README.md               # Project documentation
├── src/
│   ├── __init__.py
│   ├── utils.py            # Core DIP algorithms (Registration, Differencing, Thresholding, Cleaning)
│   └── pipeline.py         # Standalone CLI processing pipeline
├── templates/
│   ├── index.html          # Web upload form
│   └── result.html         # Web result view and download
├── data/                   # Sample satellite imagery
├── uploads/                # Temporary user uploads
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

### Option A: Run the Web Application
Start the Flask web server:
```powershell
python app.py
```
Open your browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)** (or `http://localhost:5000`)

1. Upload **Image 1 (Date 1 / Before)**.
2. Upload **Image 2 (Date 2 / After)**.
3. Click **Run Change Detection Pipeline** to view and download the result.

---

### Option B: Run via Command-Line (CLI)
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
         Absolute Image Differencing
                   │
                   ▼
         Otsu Automatic Thresholding
                   │
                   ▼
  Morphological Filtering & Component Cleanup
                   │
                   ▼
    Side-by-Side Change Map Visualization
```

---

## 👤 Author
- **Student**: Asif Ali
- **Department**: Department of Artificial Intelligence
- **Institution**: Aror University of Art, Architecture, Design & Heritage, Sukkur, Pakistan
- **Course**: Digital Image Processing (DIP)
