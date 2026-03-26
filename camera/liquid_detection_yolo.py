"""
Liquid Detection using YOLO-NAS Model

Uses the BDD-G trained YOLO-NAS model to detect tips and liquid in pipette
tip images. Much more accurate than manual ROI + color detection.

Classes:
  - "tips":   bounding box around each pipette tip
  - "liquid": bounding box around liquid inside tips

Logic:
  - If "liquid" detected → liquid present
  - Liquid area / tip area → fill ratio (how full the tip is)
  - Per-tip analysis by matching liquid boxes to tip boxes

Usage:
    python liquid_detection_yolo.py --images ./captured_images --output ./analysis_yolo

    # Adjust confidence threshold
    python liquid_detection_yolo.py --images ./captured_images --output ./analysis_yolo --conf 0.3

    # Also run green color detection as backup
    python liquid_detection_yolo.py --images ./captured_images --output ./analysis_yolo --green
"""

import cv2
import numpy as np
import os
import re
import csv
import argparse
from pathlib import Path


# ── YOLO-NAS Model ───────────────────────────────────────────────────────

CLASS_NAMES = ["tips", "liquid"]
DEFAULT_WEIGHTS = os.path.expanduser(
    "~/autolabproject/YOLO/OT2-Computer-Vision/Trained Models_NAS/ckpt_best.pth"
)


def load_model(checkpoint_path: str):
    """Load YOLO-NAS model with trained weights."""
    from super_gradients.training import models as sg_models

    print(f"[MODEL] Loading YOLO-NAS from: {checkpoint_path}")
    model = sg_models.get(
        "yolo_nas_l",
        num_classes=len(CLASS_NAMES),
        checkpoint_path=checkpoint_path,
    )
    model.eval()
    print("[MODEL] Loaded OK.")
    return model


def predict(model, frame: np.ndarray, conf: float = 0.35):
    """
    Run inference on a BGR frame.
    Returns list of dicts: {label, conf, box: (x1, y1, x2, y2)}
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb, conf=conf)

    detections = []
    pred = results.prediction
    if pred is None:
        return detections

    boxes = pred.bboxes_xyxy
    scores = pred.confidence
    labels = pred.labels

    for box, score, label_idx in zip(boxes, scores, labels):
        label_idx = int(label_idx)
        if label_idx < len(CLASS_NAMES):
            detections.append({
                "label": CLASS_NAMES[label_idx],
                "conf": float(score),
                "box": tuple(int(v) for v in box),
            })
    return detections


# ── Analysis Helpers ─────────────────────────────────────────────────────

def box_area(box):
    """Area of a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou(box_a, box_b):
    """Intersection over union of two boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def box_overlap_ratio(inner_box, outer_box):
    """How much of inner_box is inside outer_box."""
    x1 = max(inner_box[0], outer_box[0])
    y1 = max(inner_box[1], outer_box[1])
    x2 = min(inner_box[2], outer_box[2])
    y2 = min(inner_box[3], outer_box[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = box_area(inner_box)
    return inter / inner_area if inner_area > 0 else 0.0


def match_liquid_to_tips(detections):
    """
    Match liquid detections to tip detections.
    Returns list of tip dicts with matched liquid info.
    """
    tips = sorted(
        [d for d in detections if d["label"] == "tips"],
        key=lambda d: d["box"][0],  # Sort left to right
    )
    liquids = [d for d in detections if d["label"] == "liquid"]

    tip_results = []
    for i, tip in enumerate(tips):
        tip_info = {
            "tip_num": i + 1,
            "tip_box": tip["box"],
            "tip_conf": tip["conf"],
            "tip_area": box_area(tip["box"]),
            "has_liquid": False,
            "liquid_conf": 0.0,
            "liquid_area": 0,
            "fill_ratio": 0.0,
            "liquid_height_ratio": 0.0,
        }

        # Find liquid boxes that overlap with this tip
        best_overlap = 0.0
        best_liquid = None
        for liq in liquids:
            overlap = box_overlap_ratio(liq["box"], tip["box"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_liquid = liq

        if best_liquid and best_overlap > 0.3:
            liq_box = best_liquid["box"]
            tip_box = tip["box"]

            # Clip liquid box to tip box
            clipped_x1 = max(liq_box[0], tip_box[0])
            clipped_y1 = max(liq_box[1], tip_box[1])
            clipped_x2 = min(liq_box[2], tip_box[2])
            clipped_y2 = min(liq_box[3], tip_box[3])
            clipped_area = max(0, clipped_x2 - clipped_x1) * max(0, clipped_y2 - clipped_y1)

            tip_height = tip_box[3] - tip_box[1]
            liquid_height = clipped_y2 - clipped_y1

            tip_info["has_liquid"] = True
            tip_info["liquid_conf"] = best_liquid["conf"]
            tip_info["liquid_area"] = clipped_area
            tip_info["fill_ratio"] = clipped_area / tip_info["tip_area"] if tip_info["tip_area"] > 0 else 0.0
            tip_info["liquid_height_ratio"] = liquid_height / tip_height if tip_height > 0 else 0.0

        tip_results.append(tip_info)

    return tip_results


def detect_green_in_box(frame, box, hue_low=35, hue_high=85, sat_min=40, val_min=40):
    """Detect green pixels within a bounding box. Backup method."""
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([hue_low, sat_min, val_min]),
                       np.array([hue_high, 255, 255]))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return np.count_nonzero(mask) / mask.size


# ── Image Categorization ────────────────────────────────────────────────

def categorize_images(image_dir):
    """Sort images by action type from camera_server filenames."""
    categories = {
        "pick_up_tip": [],
        "before_aspirate": [],
        "after_aspirate": [],
        "before_dispense": [],
        "after_dispense": [],
        "drop_tip": [],
    }

    files = sorted(os.listdir(image_dir))
    for f in files:
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        filepath = os.path.join(image_dir, f)
        for cat in categories:
            if cat in f.lower():
                categories[cat].append(filepath)
                break

    for cat, files_list in categories.items():
        if files_list:
            print(f"  {cat}: {len(files_list)} images")
    return categories


def extract_column(filepath):
    """Extract column from filename (e.g., 'col_A3' -> 'A3')."""
    match = re.search(r'col_([A-H]\d+)', os.path.basename(filepath))
    return match.group(1) if match else "unknown"


# ── Annotation Drawing ──────────────────────────────────────────────────

def draw_annotated(frame, detections, tip_results, action):
    """Draw bounding boxes and labels on the frame."""
    out = frame.copy()

    # Draw all detection boxes
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        if det["label"] == "tips":
            color = (0, 220, 255)  # Yellow
        else:
            color = (0, 255, 100)  # Green
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{det['label']} {det['conf']:.0%}"
        cv2.putText(out, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Draw per-tip fill info
    for tip in tip_results:
        x1, y1, x2, y2 = tip["tip_box"]
        cx = (x1 + x2) // 2
        fill = tip["fill_ratio"] * 100

        if tip["has_liquid"]:
            label = f"T{tip['tip_num']}: {fill:.0f}%"
            color = (0, 0, 255)  # Red = liquid
        else:
            label = f"T{tip['tip_num']}: empty"
            color = (200, 200, 200)  # Gray = empty

        cv2.putText(out, label, (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Status bar
    tips_with_liquid = sum(1 for t in tip_results if t["has_liquid"])
    total_tips = len(tip_results)
    status = f"{action} | {tips_with_liquid}/{total_tips} tips with liquid"
    cv2.rectangle(out, (0, 0), (400, 30), (0, 0, 0), -1)
    cv2.putText(out, status, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return out


# ── Main Analysis ───────────────────────────────────────────────────────

def analyze(image_dir, output_dir, weights_path, conf=0.35, use_green=False):
    """Full analysis pipeline using YOLO-NAS."""

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load model
    model = load_model(weights_path)

    # 2. Categorize images
    print("\n[1] Categorizing images...")
    categories = categorize_images(image_dir)

    # 3. Analyze
    print(f"\n[2] Analyzing images (conf={conf})...")
    results = []

    analysis_categories = ["before_aspirate", "after_aspirate", "before_dispense", "after_dispense"]

    for cat in analysis_categories:
        for filepath in categories[cat]:
            img = cv2.imread(filepath)
            if img is None:
                continue

            column = extract_column(filepath)

            # Run YOLO
            detections = predict(model, img, conf)
            tip_results = match_liquid_to_tips(detections)

            num_tips = len(tip_results)
            tips_with_liquid = sum(1 for t in tip_results if t["has_liquid"])

            row = {
                "filename": os.path.basename(filepath),
                "action": cat,
                "column": column,
                "tips_detected": num_tips,
                "tips_with_liquid": tips_with_liquid,
                "liquid_detected": "YES" if tips_with_liquid > 0 else "NO",
            }

            # Per-tip data
            for t in tip_results:
                n = t["tip_num"]
                row[f"tip_{n}_liquid"] = "YES" if t["has_liquid"] else "NO"
                row[f"tip_{n}_fill"] = f"{t['fill_ratio']:.4f}"
                row[f"tip_{n}_liquid_conf"] = f"{t['liquid_conf']:.4f}"
                row[f"tip_{n}_height"] = f"{t['liquid_height_ratio']:.4f}"

                # Optional green backup
                if use_green and t["tip_area"] > 0:
                    green_ratio = detect_green_in_box(img, t["tip_box"])
                    row[f"tip_{n}_green"] = f"{green_ratio:.4f}"

            # Summary
            if tip_results:
                avg_fill = np.mean([t["fill_ratio"] for t in tip_results])
                max_fill = max(t["fill_ratio"] for t in tip_results)
                row["avg_fill"] = f"{avg_fill:.4f}"
                row["max_fill"] = f"{max_fill:.4f}"
            else:
                row["avg_fill"] = "0.0000"
                row["max_fill"] = "0.0000"

            results.append(row)

            # Save annotated image
            annotated = draw_annotated(img, detections, tip_results, cat)
            ann_path = os.path.join(output_dir, f"yolo_{os.path.basename(filepath)}")
            cv2.imwrite(ann_path, annotated)

            status = f"🟢 LIQUID ({tips_with_liquid}/{num_tips})" if tips_with_liquid > 0 else f"⚪ empty (0/{num_tips})"
            print(f"  {cat:20s} | col {column:4s} | tips={num_tips} | {status}")

    # 4. Save CSV
    if results:
        csv_path = os.path.join(output_dir, "liquid_analysis_yolo.csv")
        # Collect all possible fieldnames across all rows
        all_fields = []
        seen = set()
        for row in results:
            for key in row:
                if key not in seen:
                    all_fields.append(key)
                    seen.add(key)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, restval="")
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[3] Results saved to: {csv_path}")

    # 5. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    expected = {
        "before_aspirate": "NO",
        "after_aspirate": "YES",
        "before_dispense": "YES",
        "after_dispense": "NO",
    }

    for cat in analysis_categories:
        cat_results = [r for r in results if r["action"] == cat]
        if cat_results:
            liquid_count = sum(1 for r in cat_results if r["liquid_detected"] == "YES")
            total = len(cat_results)
            avg_fill = np.mean([float(r["avg_fill"]) for r in cat_results])
            print(f"  {cat:20s}: {liquid_count}/{total} with liquid | avg_fill={avg_fill:.4f}")

    print("\nExpected vs Detected:")
    correct = 0
    total = 0
    for cat in analysis_categories:
        cat_results = [r for r in results if r["action"] == cat]
        for r in cat_results:
            total += 1
            match = r["liquid_detected"] == expected[cat]
            if match:
                correct += 1
            symbol = "✓" if match else "✗"
            print(f"  {symbol} {r['filename']}: expected={expected[cat]}, got={r['liquid_detected']}")

    print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.1f}%)" if total > 0 else "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Liquid detection using YOLO-NAS")
    parser.add_argument("--images", default="./captured_images",
                        help="Directory with captured images")
    parser.add_argument("--output", default="./analysis_yolo",
                        help="Output directory")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Path to ckpt_best.pth")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold (default: 0.35)")
    parser.add_argument("--green", action="store_true",
                        help="Also run green color detection as backup")
    args = parser.parse_args()

    analyze(args.images, args.output, args.weights, args.conf, args.green)
