"""
utils/inference.py
==================
Shared inference utilities for InfraSight AI.

Replaces InitialClassifier.h5 with the dual-output pipeline:
  • YOLOv8  → bounding boxes + light status  (Light_On / Light_Off)
  • MobileNetV2 → scene condition             (Daylight / Twilight / Night)

All functions are stateless and importable from any Streamlit page.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
import streamlit as st
import tensorflow as tf
from ultralytics import YOLO


# PATHS  
_UTILS_DIR   = Path(__file__).resolve().parent
_PROJECT_DIR = _UTILS_DIR.parent
_MODELS_DIR  = _PROJECT_DIR / "models"

YOLO_WEIGHTS_PATH = _MODELS_DIR / "streetlight_yolo" / "weights" / "best.pt"
CLF_WEIGHTS_PATH  = _MODELS_DIR / "condition_classifier_final.h5"


# LABEL MAPS
CONDITION_CLASSES = ["Daylight", "Night", "Twilight"]  
LIGHT_LABEL_MAP   = {0: "Light_Off", 1: "Light_On"}  

# Fault logic: light should be ON at Night/Twilight, OFF at Daylight
FAULT_RULES = {
    ("Night",    "Light_Off"): "⚠️  LAMP OUT",
    ("Twilight", "Light_Off"): "⚠️  LAMP OUT",
    ("Daylight", "Light_On") : "💡 LEFT ON (energy waste)",
}


# MODEL LOADING 
@st.cache_resource(show_spinner="Loading detection model…")
def load_yolo_model() -> YOLO:
    """Load YOLOv8 streetlight detection model."""
    if not YOLO_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"YOLOv8 weights not found at {YOLO_WEIGHTS_PATH}. "
            "Place best.pt in models/streetlight_yolo/weights/."
        )
    return YOLO(str(YOLO_WEIGHTS_PATH))


@st.cache_resource(show_spinner="Loading condition classifier…")
def load_condition_classifier() -> tf.keras.Model:
    """Load MobileNetV2 scene-condition classifier."""
    if not CLF_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Condition classifier not found at {CLF_WEIGHTS_PATH}. "
            "Place condition_classifier_final.h5 in models/."
        )
    return tf.keras.models.load_model(str(CLF_WEIGHTS_PATH))


# PREPROCESSING
def preprocess_for_classifier(frame_bgr: np.ndarray, img_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resize and normalise a BGR frame for the MobileNetV2 classifier.

    Parameters
    ----------
    frame_bgr : np.ndarray  HxWx3 BGR image (as returned by cv2.read / cap.read)
    img_size  : (W, H) tuple for the classifier input

    Returns
    -------
    np.ndarray  shape (1, H, W, 3) float32 in [0, 1]
    """
    rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, img_size).astype("float32") / 255.0
    return np.expand_dims(resized, axis=0)


def preprocess_for_yolo(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Return a copy of the frame ready to pass directly to YOLO.
    YOLO handles its own resizing internally; we just ensure it is BGR uint8.
    """
    if frame_bgr.dtype != np.uint8:
        frame_bgr = (frame_bgr * 255).astype(np.uint8)
    return frame_bgr



# PREDICTION — LIGHT STATUS  (YOLOv8)
def predict_status(
    frame_bgr: np.ndarray,
    yolo_model: YOLO,
    conf_threshold: float = 0.25,
) -> list[dict]:
    """
    Run YOLOv8 on a single frame and return per-detection results.

    Returns
    -------
    list of dicts, each with:
        bbox       : [x1, y1, x2, y2]  (pixel coords)
        light_label: "Light_On" | "Light_Off"
        confidence : float
    """
    ready   = preprocess_for_yolo(frame_bgr)
    results = yolo_model(ready, conf=conf_threshold, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = results.names[cls_id]   # uses YOLO's own names from data.yaml
        detections.append({
            "bbox"       : [x1, y1, x2, y2],
            "light_label": label,
            "confidence" : round(conf, 3),
        })
    return detections



# PREDICTION — SCENE CONDITION  (MobileNetV2)
def predict_condition(
    frame_bgr: np.ndarray,
    clf_model: tf.keras.Model,
    class_names: list[str] = CONDITION_CLASSES,
) -> dict:
    """
    Classify the overall scene lighting condition for a frame.

    Returns
    -------
    dict with:
        condition  : "Daylight" | "Twilight" | "Night"
        confidence : float
        probs      : dict[str, float]  — per-class softmax probabilities
    """
    batch  = preprocess_for_classifier(frame_bgr)
    probs  = clf_model.predict(batch, verbose=0)[0]
    idx    = int(np.argmax(probs))
    return {
        "condition" : class_names[idx],
        "confidence": round(float(probs[idx]), 3),
        "probs"     : {cls: round(float(p), 3) for cls, p in zip(class_names, probs)},
    }



# COMBINED INFERENCE  (single image / frame)
def predict_frame(
    frame_bgr: np.ndarray,
    yolo_model: YOLO,
    clf_model: tf.keras.Model,
    conf_threshold: float = 0.25,
) -> dict:
    """
    Run both models on one frame and return a unified result dict.

    Returns
    -------
    {
        detections : list[dict]   — from predict_status()
        condition  : str
        cond_conf  : float
        cond_probs : dict[str, float]
        fault      : str          — fault label for the dominant detection
    }
    """
    detections    = predict_status(frame_bgr, yolo_model, conf_threshold)
    cond_result   = predict_condition(frame_bgr, clf_model)

    condition = cond_result["condition"]

    # Derive overall fault status from the highest-confidence detection
    fault = "✅ Normal"
    if detections:
        top_det = max(detections, key=lambda d: d["confidence"])
        fault   = FAULT_RULES.get((condition, top_det["light_label"]), "✅ Normal")

    return {
        "detections": detections,
        "condition" : condition,
        "cond_conf" : cond_result["confidence"],
        "cond_probs": cond_result["probs"],
        "fault"     : fault,
    }



# OCR  (Timestamp Camera overlay extraction)
def extract_ocr_data(frame_bgr: np.ndarray) -> dict:
    """
    Extract GPS coordinates and timestamp burned into the bottom-right corner
    of a Timestamp Camera recording.

    Returns
    -------
    dict with keys: date (str|None), time (str|None), lat (float|None), lon (float|None)
    """
    h, w = frame_bgr.shape[:2]
    roi  = frame_bgr[int(0.80 * h):h, int(0.50 * w):w]

    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Allow tesseract path override via env variable for cross-platform deployment
    tess_cmd = os.environ.get("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if os.path.exists(tess_cmd):
        pytesseract.pytesseract.tesseract_cmd = tess_cmd

    text = pytesseract.image_to_string(bw, config="--psm 6").replace("\n", " ")

    dt_match  = re.search(r"(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}:\d{2})", text)
    gps_match = re.search(r"([+-]?\d+\.\d+)\s*,\s*([+-]?\d+\.\d+)", text)

    return {
        "date": dt_match.group(1)            if dt_match  else None,
        "time": dt_match.group(2)            if dt_match  else None,
        "lat" : float(gps_match.group(1))    if gps_match else None,
        "lon" : float(gps_match.group(2))    if gps_match else None,
    }



# VIDEO PROCESSING  (automated inspection pipeline)
def process_video(
    video_file,                 
    yolo_model: YOLO,
    clf_model: tf.keras.Model,
    conf_threshold: float = 0.25,
    progress_bar=None,           
) -> list[dict]:
    """
    Process an uploaded Timestamp Camera video end-to-end.

    For every sampled frame (2 fps) the pipeline:
      1. OCR  → extract GPS + timestamp from overlay
      2. YOLOv8 → detect streetlights + light status
      3. MobileNetV2 → classify scene condition
      4. Deduplicate by GPS location (5 decimal places ≈ 1 m precision)

    Returns
    -------
    list of result dicts, each with:
        frame       : np.ndarray  (BGR)
        lat, lon    : float
        time        : str  "DD/MM/YYYY HH:MM:SS"
        detections  : list[dict]
        condition   : str
        cond_conf   : float
        cond_probs  : dict
        fault       : str
        # convenience scalars for CSV logging:
        light_label : str   (dominant detection label, or "Unknown")
        confidence  : float (dominant detection confidence, or 0.0)
    """
    results        = []
    seen_locations: set[tuple] = set()

   
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    cap        = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps        = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_skip = max(1, fps // 15)   # sample ~15 frames per second
    frame_id   = 0

    with st.spinner("Analysing video — please wait…"):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frame_skip == 0:

                # 1. OCR
                ocr = extract_ocr_data(frame)
                lat, lon = ocr["lat"], ocr["lon"]

                if lat is None or lon is None:
                    frame_id += 1
                    continue

                # 2. Deduplication
                loc_key = (round(lat, 5), round(lon, 5))
                if loc_key in seen_locations:
                    frame_id += 1
                    continue
                seen_locations.add(loc_key)

                result = predict_frame(frame, yolo_model, clf_model, conf_threshold)

                # 4. Convenience scalars for the CSV / UI
                top_det = (
                    max(result["detections"], key=lambda d: d["confidence"])
                    if result["detections"] else None
                )

                entry = {
                    "frame"      : frame,
                    "lat"        : lat,
                    "lon"        : lon,
                    "time"       : f"{ocr['date']} {ocr['time']}",
                    "detections" : result["detections"],
                    "condition"  : result["condition"],
                    "cond_conf"  : result["cond_conf"],
                    "cond_probs" : result["cond_probs"],
                    "fault"      : result["fault"],
                    "light_label": top_det["light_label"] if top_det else "Unknown",
                    "confidence" : top_det["confidence"]  if top_det else 0.0,
                    # Legacy field kept for dashboard compatibility
                    "status"     : (
                        "faulty" if (top_det and top_det["light_label"] == "Light_Off") else "serviceable"
                    ),
                    "lighting"   : result["condition"],
                }
                results.append(entry)

                if progress_bar is not None:
                    progress_bar.progress(min(frame_id / total_frames, 1.0))

            frame_id += 1

    cap.release()
    try:
        os.unlink(video_path)
    except OSError:
        pass

    return results
