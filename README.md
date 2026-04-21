# InfraSight AI — Streetlight Fault Detection System

A dual-model Streamlit application for automated streetlight inspection.

---

## Project Structure

```
infrasight/
│
├── models/                          ← Trained model weights (git-ignored)
│   ├── streetlight_yolo/
│   │   └── weights/
│   │       └── best.pt              ← YOLOv8 detection weights
│   ├── condition_classifier_final.h5  ← MobileNetV2 scene classifier
│   └── InitialClassifier.h5  ← Original previously used classifier
│
├── utils/                           ← Shared logic (importable from all pages)
│   ├── __init__.py                  ← Re-exports everything
│   ├── inference.py                 ← Model loading, prediction, OCR, video pipeline
│   └── data.py                      ← CSV schema, load, append helpers
│
├── pages/
│   ├── mainDash.py                  ← Management dashboard (Overview / Manual Entry / Logs)
│   └── mobileInter.py               ← Field team interface (Automated / Manual Entry)
│
├── inspection_logs.csv              ← Auto-created on first run
├── requirements.txt
└── README.md
```

---

## Models

The application replaces the previous original `InitialClassifier.h5` with a **dual-output pipeline**:

| Model | Task | Output |
|---|---|---|
| **YOLOv8n** (`best.pt`) | Detect streetlights in frame | Bounding boxes + `Light_On` / `Light_Off` |
| **MobileNetV2** (`.h5`) | Classify scene lighting | `Daylight` / `Twilight` / `Night` |

Trained both models using the provided notebook (`streetlight_fault_detection.ipynb`), then copy the outputs here:

```bash
# After training:
cp runs/streetlight_yolo/weights/best.pt          models/streetlight_yolo/weights/best.pt
cp /content/condition_classifier_final.h5         models/condition_classifier_final.h5
```

---

## Fault Logic

| Scene Condition | Light Status | Fault Flag |
|---|---|---|
| Night or Twilight | `Light_Off` | ⚠️ LAMP OUT |
| Daylight | `Light_On` | 💡 LEFT ON (energy waste) |
| Night or Twilight | `Light_On` | ✅ Normal |

---

## Setup

```bash
pip install -r requirements.txt
```

**Tesseract OCR** must be installed for the Automated video mode:
- **Windows**: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`

Set the path via environment variable if needed:
```bash
export TESSERACT_CMD=/usr/bin/tesseract
```

---

## Running

```bash
# Dashboard (management)
streamlit run pages/mainDash.py

# Mobile interface (field teams)
streamlit run pages/mobileInter.py
```

---

## Automated Video Workflow

1. Field team records video with **Timestamp Camera** app (GPS + timestamp burned into overlay)
2. Upload `.mp4` in **mobileInter → Automated** tab
3. Pipeline samples frames at 2 fps → OCR extracts GPS → YOLOv8 detects lights → MobileNetV2 classifies scene
4. Results are deduplicated by GPS location (≈1 m precision)
5. Inspector reviews flagged faults → confirms → submits to CSV log

---

## utils/ API Reference

```python
from utils import (
    # Model loading
    load_yolo_model,              # → YOLO  (cached)
    load_condition_classifier,    # → tf.keras.Model  (cached)

    # Preprocessing
    preprocess_for_classifier,    # BGR frame → (1, 224, 224, 3) float32
    preprocess_for_yolo,          # BGR frame → BGR uint8

    # Single-output predictions
    predict_status,               # frame → list[{bbox, light_label, confidence}]
    predict_condition,            # frame → {condition, confidence, probs}

    # Combined prediction
    predict_frame,                # frame → {detections, condition, cond_conf, fault, ...}

    # OCR
    extract_ocr_data,             # frame → {date, time, lat, lon}

    # Full video pipeline
    process_video,                # UploadedFile → list[result dicts]

    # Data layer
    load_logs,                    # → pd.DataFrame
    append_entries,               # list[dict] → updated pd.DataFrame
)
```
