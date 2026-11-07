import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import utils

def main(args):
    img1 = utils.load_image(Path(args.image1))
    img2 = utils.load_image(Path(args.image2))

    if args.register:
        img2 = utils.register_images(img1, img2)

    diff = utils.compute_difference(img1, img2)
    mask = utils.apply_threshold(diff, method=args.thresh_method, value=args.thresh_value)
    mask = utils.clean_mask(mask, min_area=args.min_blob)

    # visualisation
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(img1)
    axs[0].set_title('Image 1')
    axs[1].imshow(img2)
    axs[1].set_title('Image 2')
    axs[2].imshow(diff, cmap='gray')
    axs[2].set_title('Difference')
    axs[3].imshow(mask, cmap='gray')
    axs[3].set_title('Change Mask')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    out_path = Path(args.out_dir) / 'change_detection.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Result saved to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Satellite change detection pipeline')
    parser.add_argument('image1', help='Path to earlier image')
    parser.add_argument('image2', help='Path to later image')
    parser.add_argument('--register', action='store_true', help='Run image registration')
    parser.add_argument('--thresh_method', default='otsu', choices=['otsu', 'fixed'], help='Thresholding method')
    parser.add_argument('--thresh_value', type=float, default=None, help='Fixed threshold (0-1) if using fixed method')
    parser.add_argument('--min_blob', type=int, default=30, help='Minimum pixel area for a change region')
    parser.add_argument('--out_dir', default='reports', help='Directory for output images')
    args = parser.parse_args()
    main(args)
