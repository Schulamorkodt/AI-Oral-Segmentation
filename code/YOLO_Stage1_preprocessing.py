
# ============================================================
# YOLO Stage 1 — Preprocessing Pipeline
# CBCT Tooth Image Preprocessing
# ============================================================

import os
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from tqdm import tqdm


# ============================================================
# Utility Functions
# ============================================================

def load_nifti(path):
    """Load a NIfTI (.nii or .nii.gz) CBCT scan."""
    nii = nib.load(path)
    return nii.get_fdata()


def normalize_image(img):
    """Normalize intensities to 0–1."""
    img = img.astype(np.float32)
    img -= img.min()
    img /= (img.max() + 1e-8)
    return img


def apply_clahe(img):
    """Apply adaptive histogram equalization."""
    return equalize_adapthist(img, clip_limit=0.03)


def threshold_image(img, low=0.25, high=0.95):
    """Basic intensity threshold mask for CBCT."""
    mask = (img > low) & (img < high)
    return mask.astype(np.uint8)


def preprocess_slice(slice_img):
    """Full preprocessing for a single CBCT slice."""
    norm = normalize_image(slice_img)
    clahe = apply_clahe(norm)
    thresh = threshold_image(clahe)
    return norm, clahe, thresh


# ============================================================
# Visualization Helpers
# ============================================================

def show_slice(img, title=""):
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


# ============================================================
# Main Pipeline
# ============================================================

def run_preprocessing(
    input_folder="data/imagesTr",
    output_folder="data/preprocessed",
    save=True
):
    """
    Run preprocessing over all NIfTI CBCT scans inside input_folder.
    Saves processed slices as PNG images for YOLO training.
    """

    os.makedirs(output_folder, exist_ok=True)

    scans = [f for f in os.listdir(input_folder) if f.endswith(".nii") or f.endswith(".nii.gz")]

    print(f"Found {len(scans)} CBCT scans.")
    print("Running Stage 1 preprocessing...\n")

    for scan_name in tqdm(scans):
        scan_path = os.path.join(input_folder, scan_name)

        # Load volume
        vol = load_nifti(scan_path)
        vol = np.rot90(vol)       # keep orientation consistent
        vol = np.flipud(vol)

        # Process slice-by-slice
        for idx in range(vol.shape[2]):
            slice_img = vol[:, :, idx]

            norm, clahe, thresh = preprocess_slice(slice_img)

            if save:
                out_path = os.path.join(
                    output_folder, f"{scan_name.replace('.nii.gz','').replace('.nii','')}_slice{idx}.png"
                )
                cv2.imwrite(out_path, (clahe * 255).astype(np.uint8))

    print("\nStage 1 preprocessing completed!")
    print(f"Saved processed slices to: {output_folder}")


# ============================================================
# Example Preview
# ============================================================

def preview_random_slice(input_path="data/imagesTr/example.nii.gz", idx=100):
    """
    Preview a single slice from a scan with normalization,
    CLAHE, and thresholding.
    """
    vol = load_nifti(input_path)
    vol = np.rot90(vol)
    vol = np.flipud(vol)

    slice_img = vol[:, :, idx]

    norm, clahe, thresh = preprocess_slice(slice_img)

    show_slice(slice_img, "Original Slice")
    show_slice(norm, "Normalized")
    show_slice(clahe, "CLAHE")
    show_slice(thresh, "Threshold Mask")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    run_preprocessing(
        input_folder="data/imagesTr",
        output_folder="data/preprocessed",
        save=True
    )

    # Optional: preview a slice
    # preview_random_slice()
