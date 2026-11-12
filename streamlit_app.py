import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import time
from io import BytesIO

from src import utils

# Page configuration
st.set_page_config(
    page_title="Satellite Image Change Detection | DIP",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1e293b;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-weight: 600;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_image_bytes(file_bytes: bytes) -> np.ndarray:
    return utils.load_image_from_bytes(file_bytes)


@st.cache_data(show_spinner=False)
def load_sample_images():
    sample1 = Path('data') / 'img1.jpg'
    sample2 = Path('data') / 'img2.jpg'
    if not (sample1.exists() and sample2.exists()):
        return None, None
    img1 = utils.load_image(sample1)
    img2 = utils.load_image(sample2)
    return img1, img2


@st.cache_data(show_spinner=False)
def compute_registration_and_diff(img1: np.ndarray, img2: np.ndarray, register: bool, reg_mode: str, diff_mode: str):
    t_start = time.perf_counter()
    if register:
        img2_aligned, reg_success, reg_status = utils.register_images(img1, img2, motion_type=reg_mode)
    else:
        if img1.shape[:2] != img2.shape[:2]:
            img2_aligned = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            img2_aligned = img2
        reg_success = False
        reg_status = "Bypassed (Native alignment)"

    diff = utils.compute_difference(img1, img2_aligned, method=diff_mode)
    elapsed = time.perf_counter() - t_start
    return img2_aligned, diff, reg_success, reg_status, elapsed


def main():
    st.markdown('<div class="main-title">🛰️ Satellite Image Change Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Digital Image Processing (DIP) Pipeline for Land-Cover & Urban Change Analysis</div>', unsafe_allow_html=True)

    # Sidebar: Data Source & Hyperparameters
    st.sidebar.header("📁 Data Source")
    data_source = st.sidebar.radio(
        "Choose Satellite Imagery Input",
        ["⚡ Use Preloaded Sample (Dubai Island)", "📤 Upload Custom Image Pair"],
        index=0
    )

    img1, img2 = None, None

    if data_source == "⚡ Use Preloaded Sample (Dubai Island)":
        img1, img2 = load_sample_images()
        if img1 is None:
            st.error("Sample images not found in `data/img1.jpg` and `data/img2.jpg`.")
            return
        st.sidebar.success("Sample images loaded successfully.")
    else:
        col_up1, col_up2 = st.sidebar.columns(2)
        with col_up1:
            f1 = st.file_uploader("Date 1 (Before)", type=["jpg", "jpeg", "png", "tif", "bmp"], key="file1")
        with col_up2:
            f2 = st.file_uploader("Date 2 (After)", type=["jpg", "jpeg", "png", "tif", "bmp"], key="file2")

        if f1 is not None and f2 is not None:
            try:
                img1 = load_image_bytes(f1.getvalue())
                img2 = load_image_bytes(f2.getvalue())
                st.sidebar.success("Custom images loaded successfully.")
            except Exception as e:
                st.sidebar.error(f"Error loading image: {e}")
                return
        else:
            st.info("👈 Please upload both **Date 1 (Before)** and **Date 2 (After)** images in the sidebar to begin, or select **⚡ Use Preloaded Sample**.")
            return

    # Sidebar: DIP Pipeline Controls
    st.sidebar.header("⚙️ DIP Algorithm Settings")

    with st.sidebar.expander("1. Spatial Registration (ECC)", expanded=False):
        do_register = st.checkbox("Enable ECC Auto-Alignment", value=False, help="Aligns images using Enhanced Correlation Coefficient to prevent false change from camera shifts.")
        reg_mode = st.selectbox("Motion Model", ["homography", "affine", "euclidean"], index=0)

    with st.sidebar.expander("2. Differencing & Thresholding", expanded=True):
        diff_mode = st.selectbox("Differencing Mode", ["luminance", "euclidean"], index=0, help="Luminance (perceptual grayscale) or Euclidean (RGB distance)")
        thresh_method = st.selectbox("Thresholding Method", ["otsu", "triangle", "fixed"], index=0, format_func=lambda x: "Otsu (Automatic)" if x == "otsu" else ("Triangle (Automatic)" if x == "triangle" else "Manual Slider"))
        
        thresh_value = 0.35
        if thresh_method == "fixed":
            thresh_value = st.slider("Manual Threshold Cutoff", min_value=0.01, max_value=0.99, value=0.30, step=0.01)

    with st.sidebar.expander("3. Morphological Noise Filtering", expanded=True):
        morph_op = st.selectbox("Morphological Operation", ["open_close", "open", "close", "none"], index=0, format_func=lambda x: {"open_close": "Opening & Closing", "open": "Opening (Remove Speckles)", "close": "Closing (Fill Holes)", "none": "None"}.get(x, x))
        kernel_size = st.slider("Structuring Element Size", min_value=1, max_value=9, value=3, step=2)
        min_blob = st.slider("Minimum Blob Area (px)", min_value=1, max_value=500, value=30, step=5, help="Components with area smaller than this are filtered out as noise.")

    with st.sidebar.expander("4. Display & Aesthetics", expanded=False):
        colormap = st.selectbox("Difference Colormap", ["inferno", "viridis", "jet", "turbo", "hot", "magma"], index=0)
        overlay_color = st.selectbox("Overlay Color", list(utils.OVERLAY_COLORS.keys()), index=0)
        overlay_alpha = st.slider("Overlay Transparency", min_value=0.1, max_value=1.0, value=0.55, step=0.05)
        draw_contours = st.checkbox("Draw Boundary Contours", value=True)

    # Execute Processing Pipeline
    with st.spinner("Processing satellite change detection..."):
        t0 = time.perf_counter()
        img2_aligned, diff, reg_success, reg_status, reg_time = compute_registration_and_diff(
            img1, img2, do_register, reg_mode, diff_mode
        )

        # Thresholding
        raw_mask, active_threshold = utils.apply_threshold_with_val(diff, method=thresh_method, value=thresh_value)
        _, otsu_ref = utils.apply_threshold_with_val(diff, method='otsu')

        # Morphological Cleaning
        cleaned_mask, blob_stats = utils.clean_mask_with_stats(
            raw_mask, min_area=min_blob, kernel_size=kernel_size, morph_op=morph_op
        )

        # Visuals
        diff_heatmap = utils.create_heatmap(diff, colormap=colormap)
        overlay = utils.create_overlay(
            img2_aligned, cleaned_mask, color_name=overlay_color, alpha=overlay_alpha, draw_contours=draw_contours
        )
        total_time = (time.perf_counter() - t0) * 1000

    # Calculate Metrics
    h, w = diff.shape[:2]
    total_pixels = int(h * w)
    changed_pixels = int(np.sum(cleaned_mask > 0))
    changed_pct = round((changed_pixels / total_pixels) * 100.0, 2)
    unchanged_pct = round(100.0 - changed_pct, 2)

    # Top KPI Summary Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Changed Area", f"{changed_pct}%", f"{changed_pixels:,} px")
    kpi2.metric("Unchanged Area", f"{unchanged_pct}%", f"{total_pixels - changed_pixels:,} px")
    kpi3.metric("Change Regions", f"{blob_stats['blob_count']:,} blobs", f"Max: {blob_stats['max_blob_area']:,} px")
    kpi4.metric("Cutoff Threshold", f"{active_threshold:.3f}", f"Otsu ref: {otsu_ref:.3f}")
    kpi5.metric("Processing Time", f"{total_time:.1f} ms", f"{w}x{h} px")

    # Main Tab Navigation
    tab_grid, tab_comparison, tab_analysis, tab_export = st.tabs([
        "🖼️ 4-Panel Analysis Grid",
        "🎚️ Interactive Comparison",
        "📊 DIP Histogram & Analytics",
        "💾 Export Suite"
    ])

    with tab_grid:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Date 1 (Before)")
            st.image(np.clip(img1, 0, 1), use_container_width=True)
            
            st.subheader("3. Difference Intensity Heatmap")
            st.image(diff_heatmap, use_container_width=True, caption=f"Colormap: {colormap.capitalize()}")

        with col2:
            st.subheader("2. Date 2 (After / Aligned)")
            st.image(np.clip(img2_aligned, 0, 1), use_container_width=True, caption=reg_status)

            st.subheader("4. Cleaned Binary Change Mask")
            st.image(cleaned_mask * 255, use_container_width=True, caption=f"Threshold: {active_threshold:.3f} | Min Blob: {min_blob}px")

        st.markdown("---")
        st.subheader("✨ Alpha-Blended Change Overlay")
        st.image(overlay, use_container_width=True, caption=f"Color: {overlay_color.capitalize()} | Opacity: {int(overlay_alpha * 100)}%")

    with tab_comparison:
        st.subheader("Interactive Visual Comparison")
        comp_mode = st.radio("Comparison Mode", ["Side-by-Side Synchronized", "Alpha-Blend Slider (Before ↔ After)", "Overlay Transparency Sweep"], horizontal=True)

        if comp_mode == "Side-by-Side Synchronized":
            c1, c2 = st.columns(2)
            c1.markdown("#### Date 1 (Before)")
            c1.image(np.clip(img1, 0, 1), use_container_width=True)
            c2.markdown("#### Date 2 (After)")
            c2.image(np.clip(img2_aligned, 0, 1), use_container_width=True)
        elif comp_mode == "Alpha-Blend Slider (Before ↔ After)":
            blend_val = st.slider("Blend: Date 1 (0.0) ⟷ Date 2 (1.0)", 0.0, 1.0, 0.5, 0.05)
            blended = (1.0 - blend_val) * img1 + blend_val * img2_aligned
            st.image(np.clip(blended, 0, 1), use_container_width=True, caption=f"Date 1: {int((1-blend_val)*100)}% | Date 2: {int(blend_val*100)}%")
        else:
            blend_mask = st.slider("Overlay Intensity", 0.0, 1.0, 0.6, 0.05)
            sweep_overlay = utils.create_overlay(img2_aligned, cleaned_mask, color_name=overlay_color, alpha=blend_mask, draw_contours=draw_contours)
            st.image(sweep_overlay, use_container_width=True, caption=f"Change Overlay at {int(blend_mask*100)}% Intensity")

    with tab_analysis:
        st.subheader("Difference Intensity Histogram & Otsu Cutoff")
        
        hist_data = utils.compute_histogram(diff, bins=64)
        
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        ax.bar(hist_data["bin_centers"], hist_data["counts"], width=3.5, color='#4f46e5', alpha=0.75, label='Pixel Frequency')
        
        active_thresh_val = active_threshold * 255.0
        otsu_val = otsu_ref * 255.0
        
        ax.axvline(active_thresh_val, color='#ef4444', linestyle='--', linewidth=2, label=f'Active Threshold ({active_thresh_val:.1f})')
        if abs(active_thresh_val - otsu_val) > 2.0:
            ax.axvline(otsu_val, color='#10b981', linestyle=':', linewidth=2, label=f'Otsu Reference ({otsu_val:.1f})')
            
        ax.set_title("Pixel Difference Magnitude Distribution", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Difference Value (0 - 255)", fontsize=10)
        ax.set_ylabel("Pixel Count", fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper right')
        
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### Morphological Blob Statistics")
        st.write({
            "Total Identified Change Clusters": blob_stats["blob_count"],
            "Maximum Cluster Area (pixels)": blob_stats["max_blob_area"],
            "Mean Cluster Area (pixels)": blob_stats["mean_blob_area"],
            "Registration Status": reg_status,
            "Spatial Dimensions": f"{w} x {h} pixels"
        })

    with tab_export:
        st.subheader("💾 Export & Download Suite")
        st.write("Generate and export full-resolution results for your reports or research presentations.")

        col_e1, col_e2, col_e3 = st.columns(3)

        # 1. Binary Mask PNG
        mask_uint8 = (cleaned_mask * 255).astype(np.uint8)
        _, mask_buf = cv2.imencode('.png', mask_uint8)
        col_e1.download_button(
            label="📥 Download Binary Mask (PNG)",
            data=mask_buf.tobytes(),
            file_name="change_mask.png",
            mime="image/png",
            use_container_width=True
        )

        # 2. Overlay PNG
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        _, overlay_buf = cv2.imencode('.png', overlay_bgr)
        col_e2.download_button(
            label="📥 Download Colored Overlay (PNG)",
            data=overlay_buf.tobytes(),
            file_name="change_overlay.png",
            mime="image/png",
            use_container_width=True
        )

        # 3. Full 5-panel Composite Figure
        summary_metrics = {
            "changed_percentage": changed_pct,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels
        }
        fig_bytes = utils.generate_summary_figure_bytes(
            img1, img2_aligned, diff, cleaned_mask, overlay, metrics=summary_metrics
        )
        col_e3.download_button(
            label="📥 Download 5-Panel Summary Figure",
            data=fig_bytes,
            file_name="dip_summary_figure.png",
            mime="image/png",
            use_container_width=True
        )

        st.success("All outputs can be downloaded directly in lossless PNG format.")


if __name__ == '__main__':
    main()
