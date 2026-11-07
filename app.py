import os
import time
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src import utils

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path('uploads')
app.config['RESULT_FOLDER'] = Path('static') / 'results'
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30 MB limit
app.secret_key = 'dip-project-secret-key'

# Ensure folders exist
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['RESULT_FOLDER'].mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image1' not in request.files or 'image2' not in request.files:
            flash('Both image files are required.')
            return redirect(request.url)
            
        f1 = request.files['image1']
        f2 = request.files['image2']
        
        if f1.filename == '' or f2.filename == '':
            flash('Please select both image files.')
            return redirect(request.url)
            
        if not (allowed_file(f1.filename) and allowed_file(f2.filename)):
            flash('Unsupported file format. Please upload JPG, PNG, or TIF images.')
            return redirect(request.url)
            
        # Unique timestamped filenames to prevent collisions
        ts = int(time.time() * 1000)
        ext1 = Path(f1.filename).suffix.lower()
        ext2 = Path(f2.filename).suffix.lower()
        path1 = app.config['UPLOAD_FOLDER'] / f"img1_{ts}{ext1}"
        path2 = app.config['UPLOAD_FOLDER'] / f"img2_{ts}{ext2}"
        
        f1.save(path1)
        f2.save(path2)

        try:
            # 1. Load images
            img1 = utils.load_image(path1)
            img2 = utils.load_image(path2)

            # Ensure both images have identical dimensions
            if img1.shape[:2] != img2.shape[:2]:
                import cv2
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            # 2. Registration (with safe fallback if ECC doesn't converge)
            try:
                img2_aligned = utils.register_images(img1, img2)
            except Exception as e:
                print(f"Alignment notice (fallback to original): {e}")
                img2_aligned = img2

            # 3. Differencing & Thresholding
            diff = utils.compute_difference(img1, img2_aligned)
            mask = utils.apply_threshold(diff, method='otsu')
            mask = utils.clean_mask(mask, min_area=25)

            # 4. Generate visualization
            fig, axs = plt.subplots(1, 4, figsize=(18, 5))
            axs[0].imshow(img1)
            axs[0].set_title('Date 1 (Before)', fontsize=12, fontweight='bold')
            
            axs[1].imshow(img2_aligned)
            axs[1].set_title('Date 2 (After)', fontsize=12, fontweight='bold')
            
            axs[2].imshow(diff, cmap='inferno')
            axs[2].set_title('Difference Intensity', fontsize=12, fontweight='bold')
            
            axs[3].imshow(mask, cmap='gray')
            axs[3].set_title('Detected Change Mask', fontsize=12, fontweight='bold')

            for ax in axs:
                ax.axis('off')
            plt.tight_layout()

            result_filename = f"result_{ts}.png"
            result_path = app.config['RESULT_FOLDER'] / result_filename
            plt.savefig(result_path, dpi=160, bbox_inches='tight')
            plt.close(fig)

            # Clean temporary uploads
            try:
                path1.unlink(missing_ok=True)
                path2.unlink(missing_ok=True)
            except Exception:
                pass

            return render_template('result.html', result_image=url_for('static', filename=f"results/{result_filename}"))

        except Exception as e:
            flash(f"Error processing images: {str(e)}")
            return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    # Binds to 0.0.0.0 for external/network access, but client browses localhost or 127.0.0.1
    app.run(debug=True, host='0.0.0.0', port=5000)
