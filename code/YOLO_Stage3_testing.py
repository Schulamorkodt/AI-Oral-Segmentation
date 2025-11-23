# ============================================================
# YOLO Stage 3 — Segmentation Testing & Evaluation
# ============================================================

from ultralytics import YOLO
import os
import glob
import cv2
import shutil


# ============================================================
# 1. Load Model
# ============================================================

def load_model(model_path="runs/train/YOLO_seg/weights/best.pt"):
    """
    Load a trained YOLOv8 segmentation model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model not found at {model_path}")

    print(f"[INFO] Loaded YOLO model from: {model_path}")
    return YOLO(model_path)


# ============================================================
# 2. Run Inference on a Single Image
# ============================================================

def predict_single(model, img_path, save_dir="outputs/single"):
    """
    Runs inference on one image and saves visualization.
    """
    os.makedirs(save_dir, exist_ok=True)

    results = model(img_path, save=True, project=save_dir, name="pred")

    print(f"[INFO] Saved prediction for {img_path} → {save_dir}/pred")
    return results


# ============================================================
# 3. Run Inference on a Folder of Images
# ============================================================

def predict_folder(model, folder_path, output_dir="outputs/batch"):
    """
    Runs inference on all images inside a folder.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")) +
                         glob.glob(os.path.join(folder_path, "*.jpg")) +
                         glob.glob(os.path.join(folder_path, "*.jpeg")))

    if len(image_paths) == 0:
        print(f"[WARN] No images found in {folder_path}")
        return

    print(f"[INFO] Running inference on {len(image_paths)} images...")

    model.predict(
        source=folder_path,
        save=True,
        imgsz=640,
        project=output_dir,
        name="preds",
        exist_ok=True
    )

    print(f"[INFO] Batch predictions saved to: {output_dir}/preds")


# ============================================================
# 4. Evaluate Model on Validation Set
# ============================================================

def evaluate_model(model, data_yaml="data/tooth.yaml"):
    """
    Computes YOLO segmentation metrics (mAP50/95, IoU, precision, recall).
    """
    print("[INFO] Running validation…")

    metrics = model.val(
        data=data_yaml,
        save_json=False,
        save_hybrid=False,
        plots=True
    )

    print("[INFO] Validation complete. Metrics Summary:")
    print(metrics)

    return metrics


# ============================================================
# 5. Main Execution
# ============================================================

if __name__ == "__main__":

    # STEP 1 — Load model
    model = load_model("runs/train/YOLO_seg/weights/best.pt")

    # STEP 2 — Run eval on val set
    evaluate_model(model, "data/tooth.yaml")

    # STEP 3 — Predict on a single sample image (edit path)
    # predict_single(model, "data/images/val/sample.png")

    # STEP 4 — Predict on entire val folder
    predict_folder(model, "data/images/val", output_dir="outputs")
