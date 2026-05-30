# Ceramic Defect Detection MVP

This project is a starting point for a ceramic defect detection system using Python.
It includes a live detection CLI and a simple Streamlit dashboard.

## Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run defect detection from command line

- USB camera live detection:
  ```bash
  python defect_detect.py --model yolov8n.pt
  ```
- Video file detection:
  ```bash
  python defect_detect.py path/to/video.mp4 --model yolov8n.pt
  ```
- Image file detection:
  ```bash
  python defect_detect.py path/to/image.png --model yolov8n.pt
  ```

## Run the dashboard

```bash
streamlit run streamlit_app.py
```

Open the browser link that Streamlit provides.

## Notes for production use

- Replace the default YOLO model with a defect-trained model for ceramics.
- Use consistent, diffuse lighting and a fixed camera mount.
- Capture representative examples of good ceramics and defect types.
- Common defect categories include cracks, chips, glaze problems, and contamination.

## File overview

- `defect_detect.py` — CLI runner for image/video/camera detection.
- `defect_model.py` — YOLO model wrapper and prediction helper.
- `streamlit_app.py` — simple dashboard with live stream and defect summary.
- `requirements.txt` — required Python packages.
