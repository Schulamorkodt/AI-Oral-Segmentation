# AI Oral Segmentation Using YOLOv8 + CBCT

This repository contains the full 3-stage YOLOv8 pipeline for automated tooth detection and segmentation from CBCT scans for the *Research Science Institute (RSI 2025)*.

The pipeline is structured into three stages:

### **Stage 1 — Preprocessing**  
Normalization, histogram equalization, intensity clipping, ROI extraction, and NIfTI CBCT slicing.

### **Stage 2 — Segmentation (YOLOv8)**  
Training the YOLO model on preprocessed slices, generating labels, visualizing masks, and computing IoU/Dice scores.

### **Stage 3 — Testing & Evaluation**  
Loading the trained model, performing inference on unseen CBCT scans, visualizing overlay masks, and collecting metrics.

---

## 📂 Directory Structure

```
ai-oral-segmentation/
│
├── notebooks/
│   ├── YOLO_Stage1_preprocessing_RSI2025.ipynb
│   ├── YOLO_Stage2_segmentation_RSI2025.ipynb
│   └── YOLO_Stage3_testing_RSI2025.ipynb
│
└── src/
    ├── YOLO_Stage1_preprocessing.py
    ├── YOLO_Stage2_segmentation.py
    └── YOLO_Stage3_testing.py
```

Each python file is extracted from the corresponding notebook.

---

##  Environment Setup (Conda)

```bash
conda env create -f environment.yml
conda activate ai-oral-seg
```

---

## Running the Pipeline

### **Stage 1 preprocessing**
```bash
python src/YOLO_Stage1_preprocessing.py
```

### **Stage 2 segmentation**
```bash
python src/YOLO_Stage2_segmentation.py
```

### **Stage 3 testing**
```bash
python src/YOLO_Stage3_testing.py
```

---

## Model Summary

- Framework: **Ultralytics YOLOv8**
- Data: CBCT NIfTI dental scans
- Tasks: Tooth detection, segmentation, and ROI mapping
- Metrics: IoU, Dice, Pixel Accuracy
- Stage-1 preprocessing improves segmentation IoU by **12–17%**

---

