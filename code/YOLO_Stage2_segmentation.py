# ============================================================
# YOLO Stage 2 — Segmentation Training Script
# ============================================================

from ultralytics import YOLO
import os
import shutil
import yaml


# ============================================================
# 1. Create YOLO Dataset YAML
# ============================================================

def create_yolo_yaml(
    yaml_path="data/tooth.yaml",
    train_images="data/images/train",
    val_images="data/images/val",
    train_labels="data/labels/train",
    val_labels="data/labels/val"
):
    """
    Creates the YOLO dataset definition YAML.
    """

    data_yaml = {
        "path": "data",   # root directory
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "tooth"
        }
    }

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    print(f"[INFO] YOLO dataset YAML created at: {yaml_path}")


# ============================================================
# 2. Train YOLOv8 Segmentation Model
# ============================================================

def train_yolo(
    yaml_path="data/tooth.yaml",
    model_size="yolov8n-seg.pt",
    epochs=80,
    imgsz=640,
    save_dir="runs/train/YOLO_seg"
):
    """
    Train YOLOv8 segmentation model.
    """

    print(f"[INFO] Starting YOLOv8 segmentation training...")
    print(f"[INFO] Model: {model_size}")
    print(f"[INFO] Dataset: {yaml_path}")

    model = YOLO(model_size)

    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        project="runs/train",
        name="YOLO_seg",
        save=True,
        exist_ok=True,
        verbose=True
    )

    print(f"[INFO] Training complete. Results saved to: runs/train/YOLO_seg")


# ============================================================
# 3. Optional — Visualize a Batch
# ============================================================

def visualize_batch(model_path="runs/train/YOLO_seg/weights/best.pt"):
    """
    Visualize predictions on validation set for sanity check.
    """

    model = YOLO(model_path)
    model.val(save_json=False, plots=True)

    print("[INFO] Validation plots generated.")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    # STEP 1 — Create YAML file
    create_yolo_yaml(
        yaml_path="data/tooth.yaml",
        train_images="data/images/train",
        val_images="data/images/val"
    )

    # STEP 2 — Train the YOLO segmentation model
    train_yolo(
        yaml_path="data/tooth.yaml",
        model_size="yolov8n-seg.pt",
        epochs=80,
        imgsz=640
    )

    # STEP 3 — Visualization (optional)
    # visualize_batch()
