#!/usr/bin/env python3
"""Streamlit tool to review and correct YOLO label files for defect images."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import streamlit as st


def load_class_names(config_path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    if not config_path.exists():
        return names

    in_names = False
    for line in config_path.read_text().splitlines():
        stripped = line.strip()
        if not in_names:
            if stripped.startswith("names:"):
                in_names = True
            continue
        if not line.startswith((" ", "\t")):
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key.isdigit():
            names[int(key)] = value
    return names

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def find_image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.glob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()
    )


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_labels(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []
    return [line.strip() for line in label_path.read_text().splitlines() if line.strip()]


def yolo_to_xyxy(label: str, width: int, height: int) -> tuple[int, int, int, int]:
    class_id, x_center, y_center, w, h = label.split()
    x_center, y_center, w, h = map(float, (x_center, y_center, w, h))
    x1 = int((x_center - w / 2.0) * width)
    y1 = int((y_center - h / 2.0) * height)
    x2 = int((x_center + w / 2.0) * width)
    y2 = int((y_center + h / 2.0) * height)
    return x1, y1, x2, y2


def draw_boxes(image: np.ndarray, labels: Iterable[str], class_names: dict[int, str]) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    for label in labels:
        parts = label.split()
        if len(parts) != 5:
            continue
        try:
            x1, y1, x2, y2 = yolo_to_xyxy(label, width, height)
        except ValueError:
            continue
        class_id = parts[0]
        try:
            class_name = class_names.get(int(class_id), class_id)
        except ValueError:
            class_name = class_id
        color = (255, 0, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{class_name}",
            (max(x1, 0), max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def format_labels(labels: list[str]) -> str:
    return "\n".join(labels)


def save_labels(label_path: Path, labels_text: str) -> None:
    label_path.write_text(labels_text.strip() + "\n" if labels_text.strip() else "")


def collect_image_list(data_dir: Path, split: str) -> list[tuple[Path, Path]]:
    images_dir = data_dir / "images" / split
    labels_dir = data_dir / "labels" / split
    if not images_dir.exists():
        return []
    images = find_image_files(images_dir)
    pairs = []
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        pairs.append((image_path, label_path))
    return pairs


def main() -> int:
    st.set_page_config(page_title="Label Review", layout="wide")
    st.title("YOLO Label Review & Correction")
    st.markdown(
        "Use this tool to review YOLO labels for defect images, inspect bounding boxes, and save corrected label files."
    )

    data_dir = Path(st.sidebar.text_input("Dataset base directory", "datasets/pseudo"))
    data_config = Path(st.sidebar.text_input("Dataset config file", "data_pseudo.yaml"))
    class_names = load_class_names(data_config)
    split = st.sidebar.selectbox("Split", ["train", "val"])

    image_pairs = collect_image_list(data_dir, split)
    if not image_pairs:
        st.warning("No labeled images found under the chosen dataset directory and split.")
        st.sidebar.write("Expected structure: datasets/pseudo/images/train and datasets/pseudo/labels/train")
        return 0

    if class_names:
        st.sidebar.markdown("**Class mapping**")
        for class_id, name in sorted(class_names.items()):
            st.sidebar.write(f"{class_id}: {name}")
    else:
        st.sidebar.info("No class names loaded from config. Labels will display numeric IDs.")

    filenames = [path.name for path, _ in image_pairs]
    if "label_review_index" not in st.session_state:
        st.session_state.label_review_index = 0

    if st.sidebar.button("Previous") and st.session_state.label_review_index > 0:
        st.session_state.label_review_index -= 1
    if st.sidebar.button("Next") and st.session_state.label_review_index < len(filenames) - 1:
        st.session_state.label_review_index += 1

    selected_index = st.sidebar.selectbox(
        "Select image",
        range(len(filenames)),
        format_func=lambda i: filenames[i],
        index=st.session_state.label_review_index,
        key="label_review_select",
    )
    st.session_state.label_review_index = selected_index
    image_path, label_path = image_pairs[selected_index]

    labels = load_labels(label_path)
    label_text = st.text_area(
        "YOLO label text",
        value=format_labels(labels),
        height=240,
        key=f"label_text_{selected_index}",
    )

    image = load_image(image_path)
    annotated = draw_boxes(image, label_text.splitlines(), class_names)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption=f"Original image: {image_path.name}", use_column_width=True)
    with col2:
        st.image(annotated, caption="Annotated boxes", use_column_width=True)

    if st.button("Save labels"):
        save_labels(label_path, label_text)
        st.success(f"Saved labels to {label_path}")

    st.markdown("---")
    st.write("Label file:")
    st.code(str(label_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
