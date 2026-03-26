"""
Combined Liquid Analysis — YOLO-NAS + Green Color Detection

Analyzes pipette tip images using two methods:
  1. YOLO-NAS: Detects tip and liquid bounding boxes, calculates fill ratio
  2. Green HSV: Detects green liquid by color segmentation (primary for green liquid)

Both methods run on every image. The final "liquid_detected" verdict uses both:
  - Green detection is the primary signal (more accurate for green liquid)
  - YOLO provides per-tip bounding boxes and fill ratios

Usage:
    python liquid_analysis.py --images ./captured_images
    python liquid_analysis.py --images ./captured_images --output ./analysis_result
    python liquid_analysis.py --images ./captured_images --conf 0.3 --threshold 30

Input images should follow camera_server naming convention:
    *_pick_up_tip_*.jpg     → used as empty baseline
    *_before_aspirate_*.jpg → should show empty tips
    *_after_aspirate_*.jpg  → should show liquid in tips
    *_before_dispense_*.jpg → should show liquid in tips
    *_after_dispense_*.jpg  → should show empty tips
"""

import cv2
import numpy as np
import os
import re
import csv
import argparse


# ═══════════════════════════════════════════════════════════════════════════
# YOLO-NAS MODEL
# ═══════════════════════════════════════════════════════════════════════════

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


def yolo_predict(model, frame: np.ndarray, conf: float = 0.35):
    """
    Run YOLO-NAS inference on a BGR frame.
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


# ═══════════════════════════════════════════════════════════════════════════
# YOLO BOX HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def box_area(box):
    """Area of bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_overlap_ratio(inner_box, outer_box):
    """Fraction of inner_box that is inside outer_box."""
    x1 = max(inner_box[0], outer_box[0])
    y1 = max(inner_box[1], outer_box[1])
    x2 = min(inner_box[2], outer_box[2])
    y2 = min(inner_box[3], outer_box[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = box_area(inner_box)
    return inter / inner_area if inner_area > 0 else 0.0


def match_liquid_to_tips(detections):
    """
    Match YOLO liquid detections to tip detections.
    Returns list of tip dicts sorted left-to-right with matched liquid info.
    """
    tips = sorted(
        [d for d in detections if d["label"] == "tips"],
        key=lambda d: d["box"][0],
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


# ═══════════════════════════════════════════════════════════════════════════
# GREEN COLOR DETECTION (HSV)
# ═══════════════════════════════════════════════════════════════════════════

def detect_green_in_box(frame, box, hue_low=35, hue_high=85, sat_min=40, val_min=40):
    """
    Detect green pixels within a bounding box using HSV color segmentation.
    Returns: green_ratio (0.0 = no green, 1.0 = all green)
    """
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, 0.0, -1

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([hue_low, sat_min, val_min]),
                       np.array([hue_high, 255, 255]))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    green_ratio = np.count_nonzero(mask) / mask.size

    # Find meniscus (top of green region)
    row_sums = np.sum(mask, axis=1)
    green_rows = np.where(row_sums > 0)[0]

    if len(green_rows) == 0:
        return green_ratio, 0.0, -1

    meniscus_y = green_rows[0]
    bottom_y = green_rows[-1]
    height = mask.shape[0]
    green_height_ratio = (bottom_y - meniscus_y) / height if height > 0 else 0.0

    return green_ratio, green_height_ratio, meniscus_y


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE CATEGORIZATION
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# ANNOTATED IMAGE DRAWING
# ═══════════════════════════════════════════════════════════════════════════

def draw_annotated(frame, detections, tip_results, tip_green_data, action):
    """
    Draw YOLO bounding boxes + green detection info on the frame.
    tip_green_data: list of (green_ratio, green_height) per tip
    """
    out = frame.copy()

    # Draw YOLO detection boxes
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        if det["label"] == "tips":
            color = (0, 220, 255)  # Yellow
        else:
            color = (0, 255, 100)  # Green
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{det['label']} {det['conf']:.0%}"
        cv2.putText(out, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw per-tip info below each tip
    for i, tip in enumerate(tip_results):
        x1, y1, x2, y2 = tip["tip_box"]
        fill_pct = tip["fill_ratio"] * 100
        green_pct = tip_green_data[i][0] * 100 if i < len(tip_green_data) else 0.0

        has_liquid = tip["has_liquid"] or green_pct > 2.0

        if has_liquid:
            label = f"T{tip['tip_num']}: F{fill_pct:.0f}% G{green_pct:.0f}%"
            color = (0, 0, 255)  # Red = liquid
        else:
            label = f"T{tip['tip_num']}: empty"
            color = (200, 200, 200)  # Gray

        cv2.putText(out, label, (x1 - 5, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Status bar
    tips_with_liquid_yolo = sum(1 for t in tip_results if t["has_liquid"])
    tips_with_green = sum(1 for g in tip_green_data if g[0] > 0.02)
    total_tips = len(tip_results)
    status = f"{action} | YOLO: {tips_with_liquid_yolo}/{total_tips} | Green: {tips_with_green}/{total_tips}"
    cv2.rectangle(out, (0, 0), (480, 30), (0, 0, 0), -1)
    cv2.putText(out, status, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def analyze(image_dir, output_dir, weights_path, conf=0.35, threshold=25):
    """Combined YOLO + Green analysis pipeline."""

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load YOLO model
    model = load_model(weights_path)

    # 2. Categorize images
    print("\n[1] Categorizing images...")
    categories = categorize_images(image_dir)

    # 3. Analyze each image
    print(f"\n[2] Analyzing images (yolo_conf={conf}, green_threshold={threshold})...")
    results = []

    analysis_categories = [
        "before_aspirate", "after_aspirate",
        "before_dispense", "after_dispense",
    ]

    for cat in analysis_categories:
        for filepath in categories[cat]:
            img = cv2.imread(filepath)
            if img is None:
                continue

            column = extract_column(filepath)

            # ── YOLO Detection ──
            detections = yolo_predict(model, img, conf)
            tip_results = match_liquid_to_tips(detections)

            num_tips = len(tip_results)
            yolo_liquid_count = sum(1 for t in tip_results if t["has_liquid"])

            # ── Green Detection (inside each YOLO tip box) ──
            tip_green_data = []
            for t in tip_results:
                green_ratio, green_height, _ = detect_green_in_box(img, t["tip_box"])
                tip_green_data.append((green_ratio, green_height))

            green_liquid_count = sum(1 for g in tip_green_data if g[0] > 0.02)

            # ── Build result row ──
            row = {
                "filename": os.path.basename(filepath),
                "action": cat,
                "column": column,
                "tips_detected": num_tips,
                "yolo_liquid_tips": yolo_liquid_count,
                "green_liquid_tips": green_liquid_count,
            }

            # Combined verdict: green is primary, YOLO is secondary
            avg_green = np.mean([g[0] for g in tip_green_data]) if tip_green_data else 0.0
            row["liquid_detected"] = "YES" if (avg_green > 0.02 or yolo_liquid_count > num_tips // 2) else "NO"

            # Per-tip data
            for i, t in enumerate(tip_results):
                n = t["tip_num"]
                g_ratio, g_height = tip_green_data[i] if i < len(tip_green_data) else (0.0, 0.0)

                row[f"tip_{n}_yolo_liquid"] = "YES" if t["has_liquid"] else "NO"
                row[f"tip_{n}_yolo_fill"] = f"{t['fill_ratio']:.4f}"
                row[f"tip_{n}_yolo_conf"] = f"{t['liquid_conf']:.4f}"
                row[f"tip_{n}_green_ratio"] = f"{g_ratio:.4f}"
                row[f"tip_{n}_green_height"] = f"{g_height:.4f}"

            # Summary stats
            if tip_results:
                row["avg_yolo_fill"] = f"{np.mean([t['fill_ratio'] for t in tip_results]):.4f}"
                row["max_yolo_fill"] = f"{max(t['fill_ratio'] for t in tip_results):.4f}"
            else:
                row["avg_yolo_fill"] = "0.0000"
                row["max_yolo_fill"] = "0.0000"
            row["avg_green"] = f"{avg_green:.4f}"

            results.append(row)

            # ── Save annotated image ──
            annotated = draw_annotated(img, detections, tip_results, tip_green_data, cat)
            ann_path = os.path.join(output_dir, f"annotated_{os.path.basename(filepath)}")
            cv2.imwrite(ann_path, annotated)

            # ── Print status ──
            if row["liquid_detected"] == "YES":
                status = f"LIQUID (yolo={yolo_liquid_count}/{num_tips}, green={green_liquid_count}/{num_tips})"
            else:
                status = f"empty  (yolo={yolo_liquid_count}/{num_tips}, green={green_liquid_count}/{num_tips})"
            print(f"  {cat:20s} | col {column:4s} | {status}")

    # 4. Save CSV
    if results:
        csv_path = os.path.join(output_dir, "liquid_analysis.csv")
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
        print(f"\n[3] CSV saved to: {csv_path}")

    # 5. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

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
            avg_fill = np.mean([float(r["avg_yolo_fill"]) for r in cat_results])
            avg_grn = np.mean([float(r["avg_green"]) for r in cat_results])
            print(f"  {cat:20s}: {liquid_count}/{total} liquid | yolo_fill={avg_fill:.4f} | green={avg_grn:.4f}")

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
            symbol = "+" if match else "X"
            print(f"  {symbol} {r['filename']}: expected={expected[cat]}, got={r['liquid_detected']}")

    if total > 0:
        print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.1f}%)")

    print(f"\nOutput: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Liquid analysis: YOLO-NAS + Green color detection"
    )
    parser.add_argument("--images", default="./captured_images",
                        help="Directory with captured images")
    parser.add_argument("--output", default="./analysis_result",
                        help="Output directory (default: ./analysis_result)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Path to YOLO-NAS ckpt_best.pth")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="YOLO confidence threshold (default: 0.35)")
    parser.add_argument("--threshold", type=int, default=25,
                        help="Green detection threshold (default: 25)")
    args = parser.parse_args()

    analyze(args.images, args.output, args.weights, args.conf, args.threshold)
