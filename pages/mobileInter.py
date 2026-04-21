"""
pages/mobileInter.py
====================
InfraSight AI — Maintenance Team (Mobile) Interface

Two modes:
  • Automated  — upload a Timestamp Camera video → OCR + YOLOv8 + MobileNetV2
  • Manual Entry — form-based fallback for field teams without the app
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import pandas as pd
import streamlit as st
import hydralit_components as hc
from datetime import datetime

from utils import (
    load_yolo_model,
    load_condition_classifier,
    process_video,
    append_entries,
    load_logs,
)

# PAGE CONFIG
st.set_page_config(page_title="InfraSight - Mobile", page_icon="👁️", layout="wide")
st.subheader("InfraSight AI")
st.title("Maintenance Team Interface")


# LOAD MODELS 
yolo_model = load_yolo_model()
clf_model  = load_condition_classifier()


# LOAD DATA
data_df = load_logs()


# NAVIGATION
option_data = [
    {"icon": "bi bi-cpu-fill",      "label": "Automated"},
    {"icon": "bi bi-pencil-square", "label": "Manual Entry"},
]

op = hc.option_bar(
    option_definition=option_data,
    key="MobileOption",
    horizontal_orientation=True,
)


# SESSION STATE INIT
for key in ("processed_results", "faulty_results", "serviceable_results"):
    if key not in st.session_state:
        st.session_state[key] = None


# =========================================================
# AUTOMATED MODE
# =========================================================
if op == "Automated":

    st.subheader("Automated Inspection (Video + OCR + AI)")

    # ── App instructions ─────────────────────────────────────────────────────
    st.markdown("#### 📲 Required App for Recording")
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 15px;">
            <img src="http://www.timestampcamera.com/TimestampCamera_files/icon175x175.jpeg"
                 width="50" style="border-radius: 10px;">
            <div style="line-height: 1.4;">
                Record inspection videos with the <b>Timestamp Camera App</b>
                to ensure accurate GPS and timestamp extraction via OCR.
            </div>
        </div>
        <br>
        <div style="display: flex; gap: 10px;">
            <a href="https://apps.apple.com/us/app/timestamp-camera-basic/id840110184" target="_blank">
                <img src="http://www.timestampcamera.com/iOS.png" height="40">
            </a>
            <a href="https://play.google.com/store/apps/details?id=com.jeyluta.timestampcamerafree" target="_blank">
                <img src="http://www.timestampcamera.com/Android.png" height="40">
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Upload & process
    file = st.file_uploader("Upload Timestamp Video", type=["mp4", "mov"])

    if st.button("Start Processing") and file:
        progress = st.progress(0)
        results  = process_video(file, yolo_model, clf_model, progress_bar=progress)
        progress.empty()

        if not results:
            st.warning("No valid frames detected — check that the video has visible GPS overlay.")
        else:
            st.success(f"✅ {len(results)} unique locations processed")
            st.session_state.processed_results    = results
            st.session_state.faulty_results       = [r for r in results if r["status"] == "faulty"]
            st.session_state.serviceable_results  = [r for r in results if r["status"] == "serviceable"]

    # Review & confirm
    if st.session_state.processed_results is not None:
        faulty_results      = st.session_state.faulty_results      or []
        serviceable_results = st.session_state.serviceable_results or []
        selected_indices    = []

        st.markdown("### 🔍 Fault Review")

        if not faulty_results:
            st.info("No faults detected — all lights appear serviceable. 🎉")
        else:
            for i, res in enumerate(faulty_results):
                col_img, col_info = st.columns([1, 2])

                with col_img:
                    st.image(cv2.cvtColor(res["frame"], cv2.COLOR_BGR2RGB))

                with col_info:
                    st.write(f"📍 **Lat:** {res['lat']}   **Lon:** {res['lon']}")
                    st.write(f"⏱ **Time:** {res['time']}")
                    st.write(f"💡 **Light:** {res['light_label']}  "
                             f"(conf {res['confidence']:.0%})")
                    st.write(f"🌅 **Scene:** {res['condition']}  "
                             f"(conf {res['cond_conf']:.0%})")
                    st.write(f"🚨 **Fault flag:** {res['fault']}")

                    # Per-class condition probabilities
                    with st.expander("Condition probabilities"):
                        st.json(res["cond_probs"])

                    keep = st.checkbox(
                        f"Confirm as Fault #{i + 1}", value=True, key=f"fault_{i}"
                    )
                    if keep:
                        selected_indices.append(i)

        st.divider()

        if st.button("Submit Inspection Results"):
            confirmed_faults     = [faulty_results[i] for i in selected_indices]
            final_entries        = confirmed_faults + serviceable_results

            if not final_entries:
                st.error("No entries to save — make sure the video contains valid GPS data.")
            else:
                
                log_rows = []
                for item in final_entries:
                    log_rows.append({
                        "lat"        : item["lat"],
                        "lon"        : item["lon"],
                        "status"     : item["status"],
                        "time"       : item["time"],
                        "lighting"   : item["condition"],
                        "light_label": item["light_label"],
                        "confidence" : item["confidence"],
                        "fault"      : item["fault"],
                    })

                append_entries(log_rows)
                st.success(f"✅ Logged {len(final_entries)} entries.")

                # Clear session
                for key in ("processed_results", "faulty_results", "serviceable_results"):
                    st.session_state[key] = None


# =========================================================
# MANUAL ENTRY
# =========================================================
elif op == "Manual Entry":

    st.subheader("Manual Entry")

    lat = st.number_input("Latitude",  format="%.6f")
    lon = st.number_input("Longitude", format="%.6f")
    st.markdown("[📍 Get coordinates from Google Maps](https://www.google.com/maps)")

    lighting    = st.selectbox("Lighting Condition", ["Daylight", "Twilight", "Night"])
    light_label = st.selectbox("Light Status",       ["Light_On", "Light_Off"])
    status      = "faulty" if light_label == "Light_Off" else "serviceable"

    st.info(f"Derived fault status: **{status}**")

    if st.button("Submit Entry"):
        append_entries([{
            "lat"        : lat,
            "lon"        : lon,
            "status"     : status,
            "time"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lighting"   : lighting,
            "light_label": light_label,
            "confidence" : 1.0,
            "fault"      : "Manual entry",
        }])
        st.success("Entry added successfully!")
