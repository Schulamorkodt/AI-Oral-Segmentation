# Dataset Structure — AI Oral Segmentation

This repository does **not** include the actual CBCT or YOLO images due to size and privacy constraints.  
Instead, this folder defines the expected dataset structure for running:

- Stage 1 — Preprocessing  
- Stage 2 — YOLO segmentation training  
- Stage 3 — Testing & Evaluation

---

##  Folder Layout

Your repository should contain the following folders inside `data/`:

```
data/
    images/
        train/
        val/

    labels/
        train/
        val/
```

Create these folders exactly as written.

---
##  images/train/

Place your **training images** here (PNG or JPG), e.g.:

```
0001.png
0002.png
0003.png
...
```

## images/val/

Validation images go here:

```
0120.png
0121.png
...
```

## labels/train/

Each training image must have a **YOLO segmentation label** file with the same base name:

```
0001.txt
0002.txt
0003.txt
...
```

Each line represents a segmentation mask:

```
<class_id> x1 y1 x2 y2 ... xn yn
```

- Coordinates are normalized (0–1)
- Polygons define the tooth outline
- Only one class is used: `0 = tooth`

---

## labels/val/

Matching labels for the validation images go here.

---
