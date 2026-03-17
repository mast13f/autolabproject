"""
Using BDD-G OT2-Computer-Vision YoloNAS model on your video
=============================================================

STEP 1 — Clone the repo and get the model weights
--------------------------------------------------
    git clone https://github.com/BDD-G/OT2-Computer-Vision.git

The checkpoint file you need is inside:
    OT2-Computer-Vision/Trained Models_NAS/ckpt_best.pth

STEP 2 — Install dependencies
------------------------------
    pip install super-gradients opencv-python

    Note: super-gradients requires Python 3.8-3.10.
    If you're on 3.11+ use a conda env:
        conda create -n ot2cv python=3.10
        conda activate ot2cv
        pip install super-gradients opencv-python

STEP 3 — Run this script
-------------------------
    python yolonas_annotate.py --video test.mp4 --weights path/to/ckpt_best.pth

    Or with live window:
    python yolonas_annotate.py --video test.mp4 --weights path/to/ckpt_best.pth --show
"""

import cv2
import numpy as np
import argparse
import sys
import time
from collections import deque
from pathlib import Path


# ── Class definitions from BDD-G repo ─────────────────────────────────────────
# YoloNAS model was trained with 2 classes: tips and liquid
CLASS_NAMES  = ["tips", "liquid"]
CLASS_COLORS = {
    "tips":   (0, 220, 255),   # yellow-ish
    "liquid": (0, 100, 255),   # orange
}
CONFIDENCE_THRESHOLD = 0.35


def load_model(checkpoint_path: str):
    """Load the YoloNAS model with BDD-G trained weights."""
    from super_gradients.training import models as sg_models

    print(f"Loading YoloNAS model from: {checkpoint_path}")
    # The BDD-G repo used yolo_nas_s with 2 classes
    model = sg_models.get(
        "yolo_nas_s",
        num_classes=len(CLASS_NAMES),
        checkpoint_path=checkpoint_path,
    )
    model.eval()
    print("Model loaded OK.")
    return model


def predict_frame(model, frame: np.ndarray):
    """
    Run YoloNAS inference on a single BGR frame.
    Returns list of dicts: {label, conf, box: (x1,y1,x2,y2)}
    """
    # super-gradients predict expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb, conf=CONFIDENCE_THRESHOLD)

    detections = []
    pred = results._images_prediction_lst[0]
    if pred.prediction is None:
        return detections

    boxes  = pred.prediction.bboxes_xyxy   # shape (N,4)
    scores = pred.prediction.confidence    # shape (N,)
    labels = pred.prediction.labels        # shape (N,)

    for box, score, label_idx in zip(boxes, scores, labels):
        label_idx = int(label_idx)
        if label_idx < len(CLASS_NAMES):
            detections.append({
                "label": CLASS_NAMES[label_idx],
                "conf":  float(score),
                "box":   tuple(int(v) for v in box),  # (x1,y1,x2,y2)
            })
    return detections


def draw_detections(frame: np.ndarray, detections: list, fps: float, frame_num: int) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    has_tips   = any(d["label"] == "tips"   for d in detections)
    has_liquid = any(d["label"] == "liquid" for d in detections)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        conf  = det["conf"]
        color = CLASS_COLORS[label]

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background
        text  = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # ── Status panel ──
    pw, ph = 300, 90
    panel = np.full((ph, pw, 3), 20, dtype=np.uint8)

    tip_c   = (0, 220, 255) if has_tips   else (60, 60, 60)
    liq_c   = (0, 100, 255) if has_liquid else (60, 60, 60)
    tip_txt = f"TIPS:   {'DETECTED' if has_tips   else 'not found'}"
    liq_txt = f"LIQUID: {'DETECTED' if has_liquid else 'not found'}"

    cv2.putText(panel, tip_txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tip_c,  1)
    cv2.putText(panel, liq_txt, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, liq_c,  1)
    cv2.putText(panel, f"frame {frame_num}  {fps:.1f} fps",
                (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)

    out[8:8+ph, 8:8+pw] = panel
    return out


def run(video_path: str, weights_path: str, output_path: str = None,
        show: bool = False, scale: float = 0.7):

    model = load_model(weights_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        sys.exit(1)

    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {fw}x{fh} @ {fps:.1f}fps  ({total} frames)")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))
        print(f"Saving to: {output_path}")

    fps_buf   = deque(maxlen=30)
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        t0 = time.time()

        detections = predict_frame(model, frame)
        fps_buf.append(time.time() - t0)
        fps_disp = 1.0 / (sum(fps_buf) / len(fps_buf) + 1e-9)

        annotated = draw_detections(frame, detections, fps_disp, frame_num)

        if writer:
            writer.write(annotated)

        if show:
            disp = cv2.resize(annotated, (int(fw * scale), int(fh * scale)))
            cv2.imshow("YoloNAS — Tips & Liquid", disp)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("Stopped.")
                break

        if frame_num % 100 == 0:
            pct = frame_num / max(total, 1) * 100
            labels_found = [d["label"] for d in detections]
            print(f"  [{frame_num:>6d}/{total}] {pct:.0f}%  detections: {labels_found or 'none'}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {frame_num} frames processed.")
    if output_path:
        print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BDD-G YoloNAS tips+liquid annotator")
    ap.add_argument("--video",   required=True,  help="Input video path")
    ap.add_argument("--weights", required=True,  help="Path to ckpt_best.pth")
    ap.add_argument("--output",  default=None,   help="Output video path (optional)")
    ap.add_argument("--show",    action="store_true", help="Show live window")
    ap.add_argument("--scale",   type=float, default=0.7)
    ap.add_argument("--conf",    type=float, default=0.35,
                    help="Confidence threshold (default 0.35)")
    args = ap.parse_args()

    CONFIDENCE_THRESHOLD = args.conf
    run(args.video, args.weights, args.output, args.show, args.scale)