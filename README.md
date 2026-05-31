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
  python defect_detect.py --model yolov26n.pt
  ```
- Video file detection:
  ```bash
  python defect_detect.py path/to/video.mp4 --model yolov26n.pt
  ```
- Image file detection:
  ```bash
  python defect_detect.py path/to/image.png --model yolov26n.pt
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

## Pseudo-labeling unlabeled images

If you do not yet have labeled data, you can bootstrap labels using the current defect model:

1. Put your unlabeled images in a folder such as `datasets/raw_images`.
2. Optionally capture new images directly from a camera:
   ```bash
   python capture_images.py --output-dir datasets/raw_images
   ```
   - Press `SPACE` to save a frame.
   - Press `q` or `Esc` to quit.
   - Use `--auto` and `--interval 1.0` for automatic capture.
3. Run the pseudo-label script:
   ```bash
   python pseudo_label.py --source-dir datasets/raw_images --output-dir datasets/pseudo --model yolov26n.pt --conf 0.25 --split 0.8 --save-annotated
   ```
4. Review and correct the generated `.txt` label files in:
   - `datasets/pseudo/labels/train`
   - `datasets/pseudo/labels/val`
5. Optionally use the label review UI:
   ```bash
   streamlit run label_review.py
   ```
6. Train or fine-tune with the pseudo-labeled dataset:
   ```bash
   python train.py --data data_pseudo.yaml --model yolov26n.pt --epochs 50
   ```

These scripts generate a YOLO dataset structure and a `data_pseudo.yaml` config file for training.

## Evaluate validation metrics

Once you have a trained model, evaluate it on your validation dataset.
If you do not yet have a `runs/train/.../weights/best.pt`, use a model checkpoint such as `yolov26n.pt` instead.

```bash
python evaluate.py --weights yolov26n.pt --data data_pseudo.yaml --batch 16 --imgsz 640 --plots
```

After training, replace the weights path with your trained checkpoint, for example:

```bash
python evaluate.py --weights runs/train/ceramic_defects/weights/best.pt --data data_pseudo.yaml --batch 16 --imgsz 640 --plots
```

If you use a different experiment name, replace `ceramic_defects` with the name you passed to `train.py`.

This prints:
- mean precision
- mean recall
- mAP@0.5
- mAP@0.5:0.95
- per-class precision / recall / AP
- the detection confusion matrix

If `--plots` is used, a confusion matrix image is also saved under `runs/val`.

## File overview

- `defect_detect.py` — CLI runner for image/video/camera detection.
- `defect_model.py` — YOLO model wrapper and prediction helper.
- `streamlit_app.py` — simple dashboard with live stream and defect summary.
- `capture_images.py` — helper to capture unlabeled images from camera or video.
- `pseudo_label.py` — generate pseudo-labels for YOLO training from unlabeled images.
- `label_review.py` — interactive Streamlit UI to review and correct pseudo-labels.
- `evaluate.py` — evaluate validation metrics and confusion matrices for a trained defect model.
- `train.py` — fine-tune a defect detection model with a YOLO dataset config.
- `data_pseudo.yaml` — dataset config for pseudo-labeled training data.
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
