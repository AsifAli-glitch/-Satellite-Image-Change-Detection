import os
import time
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, url_for, flash, redirect
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from io import BytesIO

from src import utils

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path('uploads')
app.config['RESULT_FOLDER'] = Path('static') / 'results'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB limit
app.secret_key = 'dip-project-satellite-change-detection'

# Ensure required directories exist
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['RESULT_FOLDER'].mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def get_or_load_images(req_data, req_files):
    """Helper to retrieve img1 and img2 from sample, session files, or direct file upload."""
    use_sample = req_data.get('use_sample', 'false').lower() == 'true'
    session_id = req_data.get('session_id', '')

    sample_img1_path = Path('data') / 'img1.jpg'
    sample_img2_path = Path('data') / 'img2.jpg'

    if use_sample:
        if not (sample_img1_path.exists() and sample_img2_path.exists()):
            raise FileNotFoundError("Sample images (data/img1.jpg, data/img2.jpg) not found on server.")
        img1 = utils.load_image(sample_img1_path)
        img2 = utils.load_image(sample_img2_path)
        return img1, img2, "sample"

    # Check for direct file uploads
    f1 = req_files.get('image1')
    f2 = req_files.get('image2')

    if f1 and f2 and f1.filename != '' and f2.filename != '':
        # Generate new session ID
        current_session = session_id or str(uuid.uuid4())
        ext1 = Path(f1.filename).suffix.lower() or '.png'
        ext2 = Path(f2.filename).suffix.lower() or '.png'
        save_path1 = app.config['UPLOAD_FOLDER'] / f"{current_session}_1{ext1}"
        save_path2 = app.config['UPLOAD_FOLDER'] / f"{current_session}_2{ext2}"
        
        f1.save(save_path1)
        f2.save(save_path2)
        
        img1 = utils.load_image(save_path1)
        img2 = utils.load_image(save_path2)
        return img1, img2, current_session

    # Check if session_id files already exist
    if session_id:
        matches1 = list(app.config['UPLOAD_FOLDER'].glob(f"{session_id}_1.*"))
        matches2 = list(app.config['UPLOAD_FOLDER'].glob(f"{session_id}_2.*"))
        if matches1 and matches2:
            img1 = utils.load_image(matches1[0])
            img2 = utils.load_image(matches2[0])
            return img1, img2, session_id

    # Fallback to sample if available
    if sample_img1_path.exists() and sample_img2_path.exists():
        img1 = utils.load_image(sample_img1_path)
        img2 = utils.load_image(sample_img2_path)
        return img1, img2, "sample"

    raise ValueError("No images provided. Please upload Image 1 and Image 2 or load sample data.")


def run_pipeline(img1: np.ndarray, img2: np.ndarray, params: dict):
    """Executes the full DIP change detection pipeline given parameters."""
    t_start = time.perf_counter()

    # 1. Image alignment / registration
    do_register = str(params.get('register', 'false')).lower() == 'true'
    reg_mode = params.get('reg_mode', 'homography')
    
    if do_register:
        img2_aligned, reg_success, reg_status = utils.register_images(img1, img2, motion_type=reg_mode)
    else:
        # Resize to match dimensions if needed
        if img1.shape[:2] != img2.shape[:2]:
            img2_aligned = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            img2_aligned = img2
        reg_success = False
        reg_status = "Registration bypassed (Raw alignment)"

    # 2. Differencing
    diff_mode = params.get('diff_mode', 'luminance')
    diff = utils.compute_difference(img1, img2_aligned, method=diff_mode)

    # 3. Thresholding
    thresh_method = params.get('thresh_method', 'otsu')
    try:
        thresh_value = float(params.get('thresh_value', 0.35))
    except (ValueError, TypeError):
        thresh_value = 0.35

    raw_mask, active_threshold = utils.apply_threshold_with_val(diff, method=thresh_method, value=thresh_value)

    # Also compute Otsu for reference
    _, otsu_ref = utils.apply_threshold_with_val(diff, method='otsu')

    # 4. Morphological cleaning & Connected Components
    try:
        min_blob = int(params.get('min_blob', 25))
    except (ValueError, TypeError):
        min_blob = 25

    try:
        kernel_size = int(params.get('kernel_size', 3))
    except (ValueError, TypeError):
        kernel_size = 3

    morph_op = params.get('morph_op', 'open_close')
    cleaned_mask, blob_stats = utils.clean_mask_with_stats(
        raw_mask, min_area=min_blob, kernel_size=kernel_size, morph_op=morph_op
    )

    # 5. Visual representations
    colormap = params.get('colormap', 'inferno')
    diff_heatmap = utils.create_heatmap(diff, colormap=colormap)

    overlay_color = params.get('overlay_color', 'sage')
    try:
        overlay_alpha = float(params.get('overlay_alpha', 0.55))
    except (ValueError, TypeError):
        overlay_alpha = 0.55

    overlay = utils.create_overlay(
        img2_aligned, cleaned_mask, color_name=overlay_color, alpha=overlay_alpha, draw_contours=True
    )

    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000, 1)

    # 6. Quantitative metrics
    h, w = diff.shape[:2]
    total_pixels = int(h * w)
    changed_pixels = int(np.sum(cleaned_mask > 0))
    changed_pct = round((changed_pixels / total_pixels) * 100.0, 2)
    unchanged_pixels = total_pixels - changed_pixels
    unchanged_pct = round(100.0 - changed_pct, 2)

    metrics = {
        "width": w,
        "height": h,
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "changed_percentage": changed_pct,
        "unchanged_pixels": unchanged_pixels,
        "unchanged_percentage": unchanged_pct,
        "blob_count": blob_stats["blob_count"],
        "max_blob_area": blob_stats["max_blob_area"],
        "mean_blob_area": blob_stats["mean_blob_area"],
        "active_threshold": round(float(active_threshold), 4),
        "otsu_threshold": round(float(otsu_ref), 4),
        "registration_applied": do_register,
        "registration_status": reg_status,
        "processing_time_ms": latency_ms
    }

    # 7. Histogram
    histogram = utils.compute_histogram(diff, bins=64)
    histogram["active_threshold"] = round(float(active_threshold * 255.0), 1)

    return {
        "img1": img1,
        "img2_aligned": img2_aligned,
        "diff": diff,
        "diff_heatmap": diff_heatmap,
        "mask": (cleaned_mask * 255).astype(np.uint8),
        "overlay": overlay,
        "metrics": metrics,
        "histogram": histogram
    }


@app.route('/', methods=['GET'])
def index():
    """Serve the interactive dashboard single-page interface."""
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def api_process():
    """Main API endpoint for real-time interactive processing."""
    try:
        # Support both JSON payload and multipart form data
        if request.is_json:
            req_data = request.get_json() or {}
            req_files = {}
        else:
            req_data = request.form.to_dict()
            req_files = request.files

        img1, img2, session_id = get_or_load_images(req_data, req_files)
        pipeline_res = run_pipeline(img1, img2, req_data)

        # Encode images to base64 Data URIs
        b64_img1 = utils.to_base64_png(pipeline_res["img1"])
        b64_img2 = utils.to_base64_png(pipeline_res["img2_aligned"])
        b64_diff = utils.to_base64_png(pipeline_res["diff_heatmap"])
        b64_mask = utils.to_base64_png(pipeline_res["mask"])
        b64_overlay = utils.to_base64_png(pipeline_res["overlay"])

        return jsonify({
            "success": True,
            "session_id": session_id,
            "images": {
                "img1": b64_img1,
                "img2": b64_img2,
                "diff_heatmap": b64_diff,
                "mask": b64_mask,
                "overlay": b64_overlay
            },
            "metrics": pipeline_res["metrics"],
            "histogram": pipeline_res["histogram"]
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route('/api/export/<export_type>', methods=['POST', 'GET'])
def api_export(export_type: str):
    """Export binary mask, overlay, or high-res 5-panel composite report."""
    try:
        req_data = request.values.to_dict()
        req_files = request.files

        img1, img2, _ = get_or_load_images(req_data, req_files)
        pipeline_res = run_pipeline(img1, img2, req_data)

        ts = int(time.time())

        if export_type == 'mask':
            success, buffer = cv2.imencode('.png', pipeline_res['mask'])
            return send_file(
                BytesIO(buffer),
                mimetype='image/png',
                as_attachment=True,
                download_name=f"satellite_change_mask_{ts}.png"
            )

        elif export_type == 'overlay':
            bgr_overlay = cv2.cvtColor(pipeline_res['overlay'], cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode('.png', bgr_overlay)
            return send_file(
                BytesIO(buffer),
                mimetype='image/png',
                as_attachment=True,
                download_name=f"satellite_change_overlay_{ts}.png"
            )

        elif export_type == 'report':
            fig_bytes = utils.generate_summary_figure_bytes(
                pipeline_res['img1'],
                pipeline_res['img2_aligned'],
                pipeline_res['diff'],
                pipeline_res['mask'],
                pipeline_res['overlay'],
                pipeline_res['metrics']
            )
            return send_file(
                BytesIO(fig_bytes),
                mimetype='image/png',
                as_attachment=True,
                download_name=f"satellite_change_report_{ts}.png"
            )

        return jsonify({"error": f"Unknown export type: {export_type}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/legacy', methods=['GET', 'POST'])
def legacy():
    """Retain original legacy interface if desired."""
    if request.method == 'POST':
        if 'image1' not in request.files or 'image2' not in request.files:
            flash('Both image files are required.')
            return redirect(request.url)
            
        f1 = request.files['image1']
        f2 = request.files['image2']
        if f1.filename == '' or f2.filename == '':
            flash('Please select both image files.')
            return redirect(request.url)

        ts = int(time.time() * 1000)
        path1 = app.config['UPLOAD_FOLDER'] / f"legacy_img1_{ts}.png"
        path2 = app.config['UPLOAD_FOLDER'] / f"legacy_img2_{ts}.png"
        f1.save(path1)
        f2.save(path2)

        img1 = utils.load_image(path1)
        img2 = utils.load_image(path2)
        pipeline_res = run_pipeline(img1, img2, {'thresh_method': 'otsu', 'min_blob': 25})
        
        result_filename = f"result_{ts}.png"
        result_path = app.config['RESULT_FOLDER'] / result_filename
        
        fig_bytes = utils.generate_summary_figure_bytes(
            pipeline_res['img1'], pipeline_res['img2_aligned'],
            pipeline_res['diff'], pipeline_res['mask'],
            pipeline_res['overlay'], pipeline_res['metrics']
        )
        with open(result_path, 'wb') as f_out:
            f_out.write(fig_bytes)

        return render_template('result.html', result_image=url_for('static', filename=f"results/{result_filename}"))

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
