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

- Replace the default YOLO model with a defect-trained model for ceramics (e.g. `yolov26n.pt`).
- Use consistent, diffuse lighting and a fixed camera mount.
- Capture representative examples of good ceramics and defect types.
- Common defect categories include cracks, chips, glaze problems, and contamination.

## File overview

- `defect_detect.py` — CLI runner for image/video/camera detection.
- `defect_model.py` — YOLO model wrapper and prediction helper.
- `streamlit_app.py` — simple dashboard with live stream and defect summary.
- `requirements.txt` — required Python packages.
 
## Publish to GitHub

1. Create a new empty repository on GitHub (via the website or `gh repo create`).

2. Add the remote and push:

```bash
cd /Users/test/Desktop/python
git remote add origin git@github.com:<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

Or, using HTTPS:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

If you have the GitHub CLI installed, you can create and push in one command:

```bash
cd /Users/test/Desktop/python
gh repo create <repo-name> --public --source=. --remote=origin --push
```

After pushing, your project will be available on GitHub and you can collaborate, open issues, or enable CI for tests.
