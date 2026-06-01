#!/usr/bin/env python3
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from defect_model import DefectModel


def find_local_models(directory: Path) -> list[str]:
    model_paths = []
    model_paths.extend(directory.glob("*.pt"))
    model_paths.extend(Path("runs/train").glob("**/*.pt"))
    return sorted(str(path.relative_to(directory)) for path in model_paths if path.is_file())


def evaluate_yolo_model(
    model_path: str,
    data_path: str,
    imgsz: int,
    batch: int,
    device: str,
    conf: float,
    iou: float,
    save_dir: Path,
) -> tuple[Any, Any]:
    model = YOLO(model_path)
    custom = {
        "data": data_path,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "conf": conf,
        "iou": iou,
        "plots": True,
        "visualize": False,
        "save_txt": False,
        "save_json": False,
        "save_conf": False,
        "task": "detect",
        "mode": "val",
    }
    args_dict = {**model.overrides, **custom}
    validator_cls = model._smart_load("validator")
    save_dir.mkdir(parents=True, exist_ok=True)
    validator = validator_cls(args=args_dict, save_dir=save_dir, _callbacks=model.callbacks)
    validator(model=model.model)
    return validator.metrics, validator.confusion_matrix


st.set_page_config(page_title="Ceramic Defect Dashboard", layout="wide")
st.title("Ceramic Defect Detection Dashboard")

models = find_local_models(Path.cwd())
default_model = models[0] if models else "yolov26n.pt"

with st.sidebar:
    st.header("Settings")
    dashboard_mode = st.selectbox("Dashboard mode", ["Detection", "Evaluation"])
    if models:
        model_choice = st.selectbox("YOLO model file", models, index=0)
        model_path = str(Path.cwd() / model_choice)
    else:
        model_path = st.text_input("YOLO model path", default_model)

    if dashboard_mode == "Detection":
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
    else:
        validation_data = st.text_input("Validation data config", "data_pseudo.yaml")
        eval_imgsz = st.number_input("Image size", min_value=64, max_value=2048, value=640, step=32)
        eval_batch = st.number_input("Batch size", min_value=1, max_value=64, value=16, step=1)
        eval_device = st.text_input("Device", "auto")
        eval_conf = st.slider("Confidence threshold", 0.0, 1.0, 0.001, 0.001)
        eval_iou = st.slider("NMS IoU threshold", 0.0, 1.0, 0.6, 0.05)
        normalize_confusion = st.checkbox("Normalize confusion matrix", value=True)
        if st.button("Evaluate"):
            st.session_state.evaluate = True

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
if "evaluate" not in st.session_state:
    st.session_state.evaluate = False
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

frame_area = st.empty()
summary_area = st.empty()
counts_area = st.empty()
history_area = st.empty()
eval_area = st.empty()

if dashboard_mode == "Evaluation":
    if st.session_state.evaluate:
        try:
            metrics, confusion = evaluate_yolo_model(
                model_path,
                validation_data,
                imgsz=eval_imgsz,
                batch=eval_batch,
                device=eval_device,
                conf=eval_conf,
                iou=eval_iou,
                save_dir=Path("runs/val"),
            )
            st.session_state.eval_results = (metrics, confusion)
            st.session_state.evaluate = False
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.session_state.evaluate = False
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
            st.session_state.evaluate = False

    if st.session_state.eval_results is not None:
        metrics, confusion = st.session_state.eval_results
        box = metrics.box

        eval_area.subheader("Validation Metrics")
        cols = st.columns(4)
        cols[0].metric("Precision", f"{box.mp:.4f}")
        cols[1].metric("Recall", f"{box.mr:.4f}")
        cols[2].metric("mAP@0.5", f"{box.map50:.4f}")
        cols[3].metric("mAP@0.5:0.95", f"{box.map:.4f}")

        class_rows = []
        names = getattr(metrics, "names", {}) or {}
        class_names = [names.get(i, str(i)) for i in range(box.nc)]
        for i, label in enumerate(class_names):
            p, r, ap50, ap = box.class_result(i)
            class_rows.append(
                {
                    "class": label,
                    "precision": f"{p:.4f}",
                    "recall": f"{r:.4f}",
                    "AP50": f"{ap50:.4f}",
                    "AP": f"{ap:.4f}",
                }
            )

        st.subheader("Per-class metrics")
        st.dataframe(class_rows)

        st.subheader("Confusion matrix")
        try:
            confusion_df = confusion.to_df()
            st.dataframe(confusion_df)
        except Exception as exc:
            st.warning(f"Could not render confusion matrix table: {exc}")
    else:
        eval_area.info("Press Evaluate to run validation on the selected dataset.")

else:
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

                    class_counts: dict[str, int] = {}
                    for defect in defects:
                        class_counts[defect["label"]] = class_counts.get(defect["label"], 0) + 1

                    if class_counts:
                        counts_area.table(
                            [{"defect_type": label, "count": count} for label, count in class_counts.items()]
                        )
                    else:
                        counts_area.info("No defects detected in this image.")

                    # Human-in-the-loop: allow accept/reject/relabel for each detection
                    st.subheader("Review detections")
                    corrected_labels = []
                    if defects:
                        cols = st.columns(len(defects))
                    else:
                        cols = st.columns(1)

                    for i, defect in enumerate(defects):
                        col = cols[i] if defects else cols[0]
                        with col:
                            st.write(f"#{i+1}: {defect['label']} ({defect['confidence']:.2f})")
                            action = st.selectbox(f"Action {i+1}", ["Accept", "Reject", "Relabel"], key=f"action_{st.session_state.total_frames}_{i}")
                            new_label = None
                            if action == "Relabel":
                                class_options = list(model.class_names.values()) if hasattr(model, "class_names") else []
                                if class_options:
                                    new_label = st.selectbox(f"New label {i+1}", class_options, key=f"relabel_{st.session_state.total_frames}_{i}")

                            if action != "Reject":
                                class_id = defect.get("class_id", 0)
                                label_to_save = new_label if new_label else defect.get("label")
                                corrected_labels.append((class_id, label_to_save, defect["bbox"]))

                    if st.button("Save corrections"):
                        from pathlib import Path
                        import json
                        save_dir = Path("datasets/hitl")
                        labels_dir = save_dir / "labels"
                        annotated_dir = save_dir / "annotated"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        labels_dir.mkdir(parents=True, exist_ok=True)
                        annotated_dir.mkdir(parents=True, exist_ok=True)

                        image_id = f"img_{st.session_state.total_frames}"
                        label_path = labels_dir / f"{image_id}.txt"
                        lines = []
                        for cid, lbl, bbox in corrected_labels:
                            if isinstance(lbl, str) and lbl in model.class_names.values():
                                rid = next((k for k, v in model.class_names.items() if v == lbl), cid)
                            else:
                                rid = cid
                            x1, y1, x2, y2 = bbox
                            h, w = annotated_frame.shape[:2]
                            x_center = (x1 + x2) / 2.0 / w
                            y_center = (y1 + y2) / 2.0 / h
                            bw = (x2 - x1) / w
                            bh = (y2 - y1) / h
                            lines.append(f"{rid} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

                        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
                        annotated_path = annotated_dir / f"{image_id}.jpg"
                        cv2.imwrite(str(annotated_path), annotated_frame)

                        log_path = save_dir / "hitl_log.jsonl"
                        entry = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "image_id": image_id,
                            "label_file": str(label_path),
                            "annotated": str(annotated_path),
                            "corrected_count": len(lines),
                        }
                        with log_path.open("a") as fh:
                            fh.write(json.dumps(entry) + "\n")

                        st.success(f"Saved {len(lines)} corrected labels to {label_path}")

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

                            class_counts: dict[str, int] = {}
                            for defect in defects:
                                class_counts[defect["label"]] = class_counts.get(defect["label"], 0) + 1

                            if class_counts:
                                counts_area.table(
                                    [{"defect_type": label, "count": count} for label, count in class_counts.items()]
                                )
                            else:
                                counts_area.info("No defects detected in this frame.")

                            # HITL controls per-frame
                            st.subheader("Review detections (live)")
                            corrected_labels = []
                            if defects:
                                cols = st.columns(len(defects))
                            else:
                                cols = st.columns(1)

                            for i, defect in enumerate(defects):
                                col = cols[i] if defects else cols[0]
                                with col:
                                    st.write(f"#{i+1}: {defect['label']} ({defect['confidence']:.2f})")
                                    action = st.selectbox(f"Action live {i+1}", ["Accept", "Reject", "Relabel"], key=f"action_live_{st.session_state.total_frames}_{i}")
                                    new_label = None
                                    if action == "Relabel":
                                        class_options = list(model.class_names.values()) if hasattr(model, "class_names") else []
                                        if class_options:
                                            new_label = st.selectbox(f"New label live {i+1}", class_options, key=f"relabel_live_{st.session_state.total_frames}_{i}")

                                    if action != "Reject":
                                        class_id = defect.get("class_id", 0)
                                        label_to_save = new_label if new_label else defect.get("label")
                                        corrected_labels.append((class_id, label_to_save, defect["bbox"]))

                            if st.button("Save live corrections"):
                                from pathlib import Path
                                import json
                                save_dir = Path("datasets/hitl")
                                labels_dir = save_dir / "labels"
                                annotated_dir = save_dir / "annotated"
                                save_dir.mkdir(parents=True, exist_ok=True)
                                labels_dir.mkdir(parents=True, exist_ok=True)
                                annotated_dir.mkdir(parents=True, exist_ok=True)

                                image_id = f"live_{st.session_state.total_frames}"
                                label_path = labels_dir / f"{image_id}.txt"
                                lines = []
                                for cid, lbl, bbox in corrected_labels:
                                    if isinstance(lbl, str) and lbl in model.class_names.values():
                                        rid = next((k for k, v in model.class_names.items() if v == lbl), cid)
                                    else:
                                        rid = cid
                                    x1, y1, x2, y2 = bbox
                                    h, w = annotated_frame.shape[:2]
                                    x_center = (x1 + x2) / 2.0 / w
                                    y_center = (y1 + y2) / 2.0 / h
                                    bw = (x2 - x1) / w
                                    bh = (y2 - y1) / h
                                    lines.append(f"{rid} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

                                label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
                                annotated_path = annotated_dir / f"{image_id}.jpg"
                                cv2.imwrite(str(annotated_path), annotated_frame)

                                log_path = save_dir / "hitl_log.jsonl"
                                entry = {
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "image_id": image_id,
                                    "label_file": str(label_path),
                                    "annotated": str(annotated_path),
                                    "corrected_count": len(lines),
                                }
                                with log_path.open("a") as fh:
                                    fh.write(json.dumps(entry) + "\n")

                                st.success(f"Saved {len(lines)} corrected labels to {label_path}")

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
