#!/usr/bin/env python3
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from defect_model import DefectModel


def find_local_models(directory: Path) -> list[str]:
    return sorted(str(path.name) for path in directory.glob("*.pt"))


st.set_page_config(page_title="Ceramic Defect Dashboard", layout="wide")
st.title("Ceramic Defect Detection Dashboard")

models = find_local_models(Path.cwd())
default_model = models[0] if models else "yolov26n.pt"

with st.sidebar:
    st.header("Settings")
    if models:
        model_choice = st.selectbox("YOLO model file", models, index=0)
        model_path = str(Path.cwd() / model_choice)
    else:
        model_path = st.text_input("YOLO model path", default_model)
    confidence = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
    source_type = st.selectbox(
        "Input source",
        ["Built-in camera", "USB camera", "Video file", "Image file"],
    )
    camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
    video_path = st.text_input("Video file path", "")
    image_path = st.text_input("Image file path", "")
    image_file = None
    if source_type == "Image file":
        image_file = st.file_uploader(
            "Upload an image file",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
        )

    if source_type == "Video file":
        st.info("Enter a local video path or use the production stream file path.")
    elif source_type == "Image file":
        st.info("Upload an image or provide a local image path.")
    else:
        st.info("Select the correct camera index for your built-in or USB camera.")

    if st.button("Start"):
        st.session_state.running = True
    if st.button("Stop"):
        st.session_state.running = False

    st.markdown("---")
    st.markdown(
        "This dashboard shows live frames, defect counts, and a short detection history. "
        "For the best results, replace the default model with a defect-trained YOLO model."
    )

if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = []
if "total_defects" not in st.session_state:
    st.session_state.total_defects = 0
if "total_frames" not in st.session_state:
    st.session_state.total_frames = 0

frame_area = st.empty()
summary_area = st.empty()
history_area = st.empty()

if st.session_state.running:
    try:
        model = DefectModel(model_path, conf=confidence, device="auto")
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.session_state.running = False
    else:
        if source_type == "Image file":
            frame = None
            if image_file is not None:
                image_bytes = image_file.read()
                np_arr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            elif image_path:
                if not Path(image_path).exists():
                    st.error("Please provide a valid image file path.")
                    st.session_state.running = False
                else:
                    frame = cv2.imread(str(Path(image_path)))
            else:
                st.error("Upload an image or provide a valid local image path.")
                st.session_state.running = False

            if frame is not None:
                annotated_frame, defects = model.predict(frame)
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_area.image(rgb_frame, caption="Image defect detection", use_column_width=True)

                defect_count = len(defects)
                st.session_state.total_frames += 1
                st.session_state.total_defects += defect_count

                summary_area.metric("Defects detected", defect_count)
                summary_area.metric("Total frames processed", st.session_state.total_frames)
                summary_area.metric("Total defects detected", st.session_state.total_defects)

                st.session_state.history.append(
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "frame": st.session_state.total_frames,
                        "defects": defect_count,
                    }
                )
                history_area.dataframe(st.session_state.history[-10:], use_container_width=True)
            st.session_state.running = False
        else:
            source = 0 if source_type == "USB camera" else video_path
            if source_type == "Video file" and not Path(video_path).exists():
                st.error("Please provide a valid video file path.")
                st.session_state.running = False
            else:
                if source_type in {"Built-in camera", "USB camera"}:
                    source = camera_index
                else:
                    source = video_path

                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    st.error(f"Unable to open source: {source}")
                    st.session_state.running = False
                else:
                    while st.session_state.running:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Stream ended or camera disconnected.")
                            break

                        annotated_frame, defects = model.predict(frame)
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        frame_area.image(rgb_frame, caption="Live defect detection", use_column_width=True)

                        defect_count = len(defects)
                        st.session_state.total_frames += 1
                        st.session_state.total_defects += defect_count

                        summary_area.metric("Defects this frame", defect_count)
                        summary_area.metric("Total frames processed", st.session_state.total_frames)
                        summary_area.metric("Total defects detected", st.session_state.total_defects)

                        st.session_state.history.append(
                            {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "frame": st.session_state.total_frames,
                                "defects": defect_count,
                            }
                        )

                        history_area.dataframe(st.session_state.history[-10:], use_container_width=True)

                        time.sleep(0.05)

                    cap.release()
else:
    frame_area.info("Press Start to begin defect detection.")
    summary_area.write(
        "Once you press Start, live frames and defect stats will appear here."
    )
