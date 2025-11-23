# AI Oral Segmentation Using YOLOv8 + CBCT

This repository contains the full 3-stage YOLOv8 pipeline for automated tooth detection and segmentation from CBCT scans

The pipeline is structured into three stages:

### **Stage 1 — Preprocessing**  
Normalization, histogram equalization, intensity clipping, ROI extraction, and NIfTI CBCT slicing.
```bash
python src/YOLO_Stage1_preprocessing.py
```

### **Stage 2 — Segmentation (YOLOv8)**  
Training the YOLO model on preprocessed slices, generating labels, visualizing masks, and computing IoU/Dice scores.
```bash
python src/YOLO_Stage2_segmentation.py
```

### **Stage 3 — Testing & Evaluation**  
Loading the trained model, performing inference on unseen CBCT scans, visualizing overlay masks, and collecting metrics.
```bash
python src/YOLO_Stage3_testing.py
```
---


##  Environment Setup (Conda)

```bash
conda env create -f environment.yml
conda activate ai-oral-seg
```

---

## Model Summary

- Framework: **Ultralytics YOLOv8**
- Data: CBCT NIfTI dental scans
- Tasks: Tooth detection, segmentation, and ROI mapping
- Metrics: IoU, Dice, Pixel Accuracy
- Stage-1 preprocessing improves segmentation IoU by **12–17%**

---

