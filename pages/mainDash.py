"""
pages/mainDash.py
=================
InfraSight AI — Maintenance Dashboard

Three tabs:
  • Overview      — map + KPI metrics + fault distribution chart
  • Manual Entry  — form entry or AI-assisted single-image prediction
  • Inspection Logs — filterable data table
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import datetime
import os
import numpy as np
import tempfile

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
import hydralit_components as hc
from PIL import Image

from utils import (
    load_yolo_model,
    load_condition_classifier,
    predict_frame,
    process_video,
    append_entries,
    load_logs,
    CONDITION_CLASSES,
)


# PAGE CONFIG
st.set_page_config(
    page_title="InfraSight - Dashboard", page_icon="👁️", layout="wide"
)
st.title("InfraSight AI — Maintenance Dashboard")


# LOAD MODELS  (cached)
yolo_model = load_yolo_model()
clf_model  = load_condition_classifier()


# NAVIGATION
option_data = [
    {"icon": "bi bi-info-circle-fill", "label": "Overview"},
    {"icon": "bi bi-file-earmark-plus-fill","label": "Submit Entry"},
    {"icon": "fa fa-database",          "label": "Inspection Logs"},
]

op = hc.option_bar(
    option_definition=option_data, key="PrimaryOption", horizontal_orientation=True
)


# LOAD DATA
data = load_logs()



# HELPERS
def get_start_date(period: str) -> datetime.datetime:
    now = datetime.datetime.now()
    return {
        "1D" : now - datetime.timedelta(days=1),
        "7D" : now - datetime.timedelta(days=7),
        "1M" : now - datetime.timedelta(days=30),
        "3M" : now - datetime.timedelta(days=90),
        "1Y" : now - datetime.timedelta(days=365),
        "YTD": datetime.datetime(now.year, 1, 1),
    }.get(period, now - datetime.timedelta(days=7))


def compute_metrics(df: pd.DataFrame) -> tuple[int, int, int]:
    total = len(df)
    working = len(df[df["status"] == "serviceable"])
    faulty  = len(df[df["status"] == "faulty"])
    return total, working, faulty


def sparkline(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series([0])
    return df.groupby(df["time"].dt.date).size()


def run_ai_on_frame(frame_bgr: np.ndarray) -> dict:
    """Thin wrapper so the UI doesn't import from utils directly."""
    return predict_frame(frame_bgr, yolo_model, clf_model)


# =========================================================
# 1. OVERVIEW
# =========================================================
if op == "Overview":
    now = datetime.datetime.now()

    col_left, col_right = st.columns([2, 1], vertical_alignment="top")

    with col_right:
        t1, s1 = st.columns([2, 1], vertical_alignment="bottom")
        with t1:
            st.subheader("Statistics")
        with s1:
            period = st.selectbox(
                "Period",
                options=["1D", "7D", "1M", "3M", "1Y", "YTD"],
                label_visibility="hidden",
                index=1,
            )

    start_date  = get_start_date(period)
    prev_start  = start_date - (now - start_date)

    current_data  = data[data["time"] >= start_date]
    previous_data = data[(data["time"] >= prev_start) & (data["time"] < start_date)]

    # Map
    with col_left:
        st.subheader(f"Streetlight Status Map ({period})")

        status_options  = ["serviceable", "faulty"]
        selected_status = st.segmented_control(
            "Filter by Status",
            options=status_options,
            default=status_options,
            selection_mode="multi",
        )

        map_data = (
            current_data[current_data["status"].isin(selected_status)]
            if selected_status
            else current_data.iloc[0:0]
        ).copy()

        map_data["marker_color"] = map_data["status"].apply(
            lambda x: [0, 255, 0, 160] if x == "serviceable" else [255, 0, 0, 160]
        )
        map_data["time_str"] = map_data["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

        st.markdown(
            """
            <style>
            .map-legend {
                position:absolute; top:20px; left:20px; z-index:999;
                background-color:rgba(255,255,255,0.4); padding:10px;
                border-radius:5px; box-shadow:0 0 5px rgba(0,0,0,.2);
                font-family:sans-serif; font-size:14px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        with st.container():
            st.markdown(
                """
                <div class="map-legend">
                    <span style="color:#00ff00;">●</span> Serviceable<br>
                    <span style="color:#ff0000;">●</span> Faulty
                </div>
                """,
                unsafe_allow_html=True,
            )

            layer = pdk.Layer(
                "ScatterplotLayer",
                map_data,
                get_position="[lon, lat]",
                get_color="marker_color",
                get_radius=30,
                pickable=True,
            )
            view_state = pdk.ViewState(
                latitude=26.0667, longitude=50.5577, zoom=9, pitch=0
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                    tooltip={
                        "html": (
                            "<b>Status:</b> {status}<br/>"
                            "<b>Light:</b> {light_label}<br/>"
                            "<b>Scene:</b> {lighting}<br/>"
                            "<b>Fault:</b> {fault}<br/>"
                            "<b>Time:</b> {time_str}"
                        ),
                        "style": {"backgroundColor": "steelblue", "color": "white"},
                    },
                )
            )


    # Statistics
    with col_right:
        total_c, working_c, faulty_c = compute_metrics(current_data)
        total_p, working_p, faulty_p = compute_metrics(previous_data)

        st.metric(
            "**Total Inspection Entries**", total_c, delta=total_c - total_p,
            chart_data=sparkline(current_data), border=True,
        )
        st.metric(
            "**Serviceable Lights**", working_c, delta=working_c - working_p,
            chart_data=sparkline(current_data[current_data["status"] == "serviceable"]),
            border=True,
        )
        st.metric(
            "**Faulty Lights**", faulty_c, delta=faulty_c - faulty_p,
            delta_color="inverse",
            chart_data=sparkline(current_data[current_data["status"] == "faulty"]),
            border=True,
        )

    
    chart_left, chart_right = st.columns(2)

    faults = current_data[current_data["status"] == "faulty"]

    with chart_left:
        st.subheader(f"Fault Distribution by Scene Condition ({period})")
        if not faults.empty and "lighting" in faults.columns:
            lighting_dist = faults.groupby("lighting").size().reset_index(name="count")
            fig = px.pie(
                lighting_dist, names="lighting", values="count", hole=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No faults recorded for {period}.")

    with chart_right:
        st.subheader(f"Fault Type Breakdown ({period})")
        if not faults.empty and "fault" in faults.columns:
            fault_dist = faults.groupby("fault").size().reset_index(name="count")
            fig2 = px.bar(
                fault_dist, x="fault", y="count",
                color="fault", labels={"fault": "Fault Type", "count": "Count"},
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info(f"No fault type data for {period}.")


# =========================================================
# 2. Submit ENTRY
# =========================================================
elif op == "Submit Entry":
 
    st.subheader("Add Inspection Entry")
 
    
    entry_mode = hc.option_bar(
        option_definition=[
            {"icon": "bi bi-pencil-fill", "label": "Manual Entry"},
            {"icon": "bi bi-cpu-fill",    "label": "AI Prediction"},
        ],
        key="ManualEntryMode",
        horizontal_orientation=False,
    )
 
    
    # SHARED HELPERS
    def time_input_widget(key_prefix: str) -> str:
        """
        Returns a timestamp string in %d/%m/%Y %H:%M:%S format.
        The user can choose 'Now' or pick a custom datetime.
        """
        use_now = st.toggle("Use current time", value=True, key=f"{key_prefix}_usenow")
        if use_now:
            ts = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            st.caption(f"🕐 {ts}")
            return ts
        else:
            col_d, col_t = st.columns(2)
            with col_d:
                picked_date = st.date_input("Date", key=f"{key_prefix}_date")
            with col_t:
                picked_time = st.time_input("Time", key=f"{key_prefix}_time",
                                            value=datetime.datetime.now().time())
            return datetime.datetime.combine(picked_date, picked_time).strftime("%d/%m/%Y %H:%M:%S")
 
    def lat_lon_inputs(key_prefix: str) -> tuple[float, float]:
        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat = st.number_input("Latitude",  format="%.6f", key=f"{key_prefix}_lat")
        with col_lon:
            lon = st.number_input("Longitude", format="%.6f", key=f"{key_prefix}_lon")
        st.markdown("[📍 Get coordinates from Google Maps](https://www.google.com/maps)")
        return lat, lon
 
    def draw_overlay(frame_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes with label + confidence onto a copy of the frame."""
        overlay = frame_bgr.copy()
        COLOURS = {"Light_On": (0, 220, 100), "Light_Off": (220, 50, 50)}
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            colour = COLOURS.get(det["light_label"], (180, 180, 180))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(
                overlay,
                f"{det['light_label']} {det['confidence']:.2f}",
                (x1, max(y1 - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2,
            )
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
 
    def submit_entry(entry: dict):
        """Append a single entry dict and show confirmation."""
        append_entries([entry])
        st.success("✅ Entry added successfully!")
 
    
    # MODE 1 — MANUAL ENTRY
    # =====================
    if entry_mode == "Manual Entry":
 
        with st.container():
            st.markdown("#### 📋 Inspection Details")

            lat, lon    = lat_lon_inputs("manual")
            lighting    = st.selectbox("Lighting Condition", ["Daylight", "Twilight", "Night"])
            light_label = st.selectbox("Light Status",       ["Light_On", "Light_Off"])
            status      = "faulty" if light_label == "Light_Off" else "serviceable"

            st.info(f"Derived fault status: **{status}**")

            st.markdown("#### 🕐 Timestamp")
            time_mode = st.radio("Time source", ["Use current time", "Pick manually"],
                                    horizontal=True)
            if time_mode == "Use current time":
                timestamp = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                st.caption(f"Will be recorded as: **{timestamp}**")
            else:
                col_d, col_t = st.columns(2)
                with col_d:
                    picked_date = st.date_input("Date", key="manual_date")
                with col_t:
                    picked_time = st.time_input("Time", key="manual_time",
                                                value=datetime.datetime.now().time())
                timestamp = datetime.datetime.combine(
                    picked_date, picked_time
                ).strftime("%d/%m/%Y %H:%M:%S")

            submitted = st.button("Submit Entry", type="primary")

        if submitted:
            submit_entry({
                "lat"        : lat,
                "lon"        : lon,
                "status"     : status,
                "time"       : timestamp,
                "lighting"   : lighting,
                "light_label": light_label,
                "confidence" : 1.0,
                "fault"      : "Manual entry",
            })
 
    
    # MODE 2 — AI PREDICTION
    # ======================
    elif entry_mode == "AI Prediction":
 
        
        st.markdown("#### 📷 Media Settings")
        has_timestamp = st.toggle(
            "Media recorded with Timestamp Camera (GPS + time burned in)",
            value=False,
        )
 
        
        with st.expander("⚙️ Model Settings", expanded=False):
            conf_threshold = st.slider(
                "YOLO Confidence Threshold", 0.05, 0.95, 0.25, 0.05,
                help="Lower = more detections, higher = fewer but more certain detections.",
            )
            iou_threshold = st.slider(
                "NMS IoU Threshold", 0.1, 0.9, 0.45, 0.05,
                help="Controls overlap suppression. Lower removes more overlapping boxes.",
            )
            clf_threshold = st.slider(
                "Condition Confidence Threshold", 0.05, 0.95, 0.50, 0.05,
                help="If the top scene-condition probability is below this, "
                     "the result is marked 'Uncertain'.",
            )
 
        # Manual GPS and time
        if not has_timestamp:
            st.markdown("#### 📍 Location & Time")
            lat, lon  = lat_lon_inputs("ai_manual")
            timestamp = time_input_widget("ai_manual")
        else:
            lat, lon, timestamp = None, None, None  
 
        
        st.markdown("#### 📁 Upload Media")
        uploaded = st.file_uploader(
            "Upload image or video",
            type=["jpg", "jpeg", "png", "mp4", "mov"],
        )
 
        if not uploaded:
            st.stop()
 
        is_video = uploaded.type in ("video/mp4", "video/quicktime")
 
        
        # IMAGE BRANCH
        if not is_video:
            image     = Image.open(uploaded).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
 
            with st.spinner("Running AI models…"):
                result = predict_frame(
                    frame_bgr, yolo_model, clf_model,
                    conf_threshold=conf_threshold,
                )
 
            detections = result["detections"]
 
            col_orig, col_overlay = st.columns(2)
            with col_orig:
                st.markdown("**Original**")
                st.image(image, use_column_width=True)
            with col_overlay:
                st.markdown("**Detection Overlay**")
                if detections:
                    st.image(draw_overlay(frame_bgr, detections), use_column_width=True)
                else:
                    st.info("No streetlights detected at this confidence threshold.")
 
            # Condition probabilities
            with st.expander("📊 Condition probabilities"):
                prob_df = pd.DataFrame(
                    result["cond_probs"].items(), columns=["Condition", "Probability"]
                ).sort_values("Probability", ascending=False)
                st.bar_chart(prob_df.set_index("Condition"))
 
            st.markdown("---")
            st.markdown("#### ✏️ Confirm & Adjust Prediction")
 
            # Editable prediction fields
            top_det = (
                max(detections, key=lambda d: d["confidence"]) if detections else None
            )
            default_light = top_det["light_label"]  if top_det else "Light_On"
            default_conf  = top_det["confidence"]   if top_det else 0.0
            predicted_cond = (
                result["condition"]
                if result["cond_conf"] >= clf_threshold
                else "Uncertain"
            )
 
            col_l, col_c = st.columns(2)
            with col_l:
                light_label = st.selectbox(
                    "Light Status",
                    ["Light_On", "Light_Off"],
                    index=0 if default_light == "Light_On" else 1,
                )
            with col_c:
                cond_options = ["Daylight", "Twilight", "Night"]
                lighting = st.selectbox(
                    "Lighting Condition",
                    cond_options,
                    index=cond_options.index(result["condition"])
                          if result["condition"] in cond_options else 0,
                )
 
            status = "faulty" if light_label == "Light_Off" else "serviceable"
            st.info(
                f"Fault status: **{status}**  |  "
                f"Model confidence: **{default_conf:.0%}**  |  "
                f"Scene confidence: **{result['cond_conf']:.0%}**"
            )
 
            if st.button("Submit Entry", type="primary", key="img_submit"):
                entry = {
                    "lat"        : lat,
                    "lon"        : lon,
                    "status"     : status,
                    "time"       : timestamp,
                    "lighting"   : lighting,
                    "light_label": light_label,
                    "confidence" : default_conf,
                    "fault"      : result["fault"],
                }
                submit_entry(entry)
 
        
        # VIDEO BRANCH
        else:
            if "video_results" not in st.session_state:
                st.session_state.video_results = None
 
            if st.button("▶ Process Video", type="primary"):
                progress = st.progress(0, text="Sampling frames…")
 
                if has_timestamp:
                    # Full OCR + dual-model
                    results = process_video(
                        uploaded, yolo_model, clf_model,
                        conf_threshold=conf_threshold,
                        progress_bar=progress,
                    )
                else:
                    # No OCR
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(uploaded.read())
                        video_path = tmp.name
 
                    cap        = cv2.VideoCapture(video_path)
                    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                    fps        = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                    skip       = max(1, fps // 2)
                    frame_id   = 0
                    results    = []
 
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_id % skip == 0:
                            r      = predict_frame(frame, yolo_model, clf_model,
                                                   conf_threshold=conf_threshold)
                            top_d  = (max(r["detections"], key=lambda d: d["confidence"])
                                      if r["detections"] else None)
                            results.append({
                                "frame"      : frame,
                                "lat"        : lat,
                                "lon"        : lon,
                                "time"       : timestamp,
                                "detections" : r["detections"],
                                "condition"  : r["condition"],
                                "cond_conf"  : r["cond_conf"],
                                "cond_probs" : r["cond_probs"],
                                "fault"      : r["fault"],
                                "light_label": top_d["light_label"] if top_d else "Unknown",
                                "confidence" : top_d["confidence"]  if top_d else 0.0,
                                "status"     : ("faulty"
                                                if top_d and top_d["light_label"] == "Light_Off"
                                                else "serviceable"),
                                "lighting"   : r["condition"],
                            })
                        progress.progress(min(frame_id / total, 1.0))
                        frame_id += 1
 
                    cap.release()
                    try:
                        os.unlink(video_path)
                    except OSError:
                        pass
 
                progress.empty()
                st.session_state.video_results = results
                st.success(f"✅ {len(results)} frames processed")
 
            # Frame review grid
            if st.session_state.video_results:
                results = st.session_state.video_results
 
                st.markdown("---")
                st.markdown(
                    "#### 🖼️ Frame Review\n"
                    "Tick frames to **include** in the submission. "
                    "Untick frames the model got wrong."
                )
 
                
                fc1, fc2, fc3 = st.columns(3)
                with fc1:
                    show_status = st.multiselect(
                        "Show status", ["faulty", "serviceable"],
                        default=["faulty", "serviceable"],
                        key="grid_status_filter",
                    )
                with fc2:
                    show_scene = st.multiselect(
                        "Show scene", ["Daylight", "Twilight", "Night"],
                        default=["Daylight", "Twilight", "Night"],
                        key="grid_scene_filter",
                    )
                with fc3:
                    min_conf = st.slider(
                        "Min detection confidence", 0.0, 1.0, 0.0, 0.05,
                        key="grid_conf_filter",
                    )
 
                visible = [
                    (i, r) for i, r in enumerate(results)
                    if r["status"] in show_status
                    and r["condition"] in show_scene
                    and r["confidence"] >= min_conf
                ]

                if not visible:
                    st.info("No frames match the current filters.")
                else:
                    COLS = 3
                    selected_indices = []

                    for row_start in range(0, len(visible), COLS):
                        cols = st.columns(COLS)
                        
                        for col_i, (global_idx, res) in enumerate(visible[row_start : row_start + COLS]):
                            
                            with cols[col_i]:
                                
                                st.image(
                                    draw_overlay(res["frame"], res["detections"])
                                    if res["detections"]
                                    else cv2.cvtColor(res["frame"], cv2.COLOR_BGR2RGB),
                                    use_column_width=True
                                )
                                
                                
                                
                                new_light = st.selectbox(
                                    "Light",
                                    ["Light_On", "Light_Off"],
                                    index=0 if res["light_label"] == "Light_On" else 1,
                                    key=f"vlight_{global_idx}",
                                )
                                
                                

                                keep = st.checkbox(
                                    "Include", value=True, key=f"vkeep_{global_idx}"
                                )
                                if keep:
                                    selected_indices.append(global_idx)
 
                    st.markdown("---")
                    st.info(
                        f"**{len(selected_indices)}** of **{len(visible)}** "
                        "visible frames selected for submission."
                    )
 
                    if st.button("Submit Selected Frames", type="primary",
                                 key="vid_submit"):
                        entries = []
                        for idx in selected_indices:
                            r = results[idx]
                            entries.append({
                                "lat"        : r["lat"],
                                "lon"        : r["lon"],
                                "status"     : r["status"],
                                "time"       : r["time"],
                                "lighting"   : r["lighting"],
                                "light_label": r["light_label"],
                                "confidence" : r["confidence"],
                                "fault"      : r["fault"],
                            })
                        append_entries(entries)
                        st.success(
                            f"✅ {len(entries)} entries logged successfully!"
                        )
                        st.session_state.video_results = None


# =========================================================
# 3. INSPECTION LOGS
# =========================================================
elif op == "Inspection Logs":

    st.subheader("Inspection Logs")

    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("Start Date")
    with col_end:
        end_date = st.date_input("End Date")

    if start_date > end_date:
        st.error("Invalid date range — start must be before end.", icon="❌")
    else:
        filtered = data.copy()
        filtered = filtered[
            (filtered["time"].dt.date >= start_date) &
            (filtered["time"].dt.date <= end_date)
        ]

        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Records",       len(filtered))
        col_b.metric("Serviceable",         len(filtered[filtered["status"] == "serviceable"]))
        col_c.metric("Faulty",              len(filtered[filtered["status"] == "faulty"]))

        
        status_filter  = st.multiselect("Filter by Status",    ["serviceable", "faulty"],  default=["serviceable", "faulty"])
        light_filter   = st.multiselect("Filter by Light",     ["Light_On", "Light_Off"],  default=["Light_On", "Light_Off"])
        scene_filter   = st.multiselect("Filter by Scene",     ["Daylight", "Twilight", "Night"], default=["Daylight", "Twilight", "Night"])

        if status_filter:
            filtered = filtered[filtered["status"].isin(status_filter)]
        if light_filter and "light_label" in filtered.columns:
            filtered = filtered[filtered["light_label"].isin(light_filter)]
        if scene_filter and "lighting" in filtered.columns:
            filtered = filtered[filtered["lighting"].isin(scene_filter)]

        st.dataframe(filtered, use_container_width=True)

        
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download filtered CSV",
            data=csv_bytes,
            file_name="inspection_logs_filtered.csv",
            mime="text/csv",
        )
