"""
Liquid Detection in Pipette Tips

Analyzes camera images to detect liquid levels in pipette tips.
Uses empty tip images as baseline, compares against aspirate/dispense images.

Approaches combined:
  1. ROI extraction — focus only on the tip region
  2. Image subtraction — compare against empty baseline
  3. Meniscus detection — find liquid level height in each tip
  4. Color analysis — detect liquid presence by color shift

Usage:
    python liquid_detection.py --images ./captured_images --output ./analysis
    python liquid_detection.py --images ./captured_images --output ./analysis --calibrate
    python liquid_detection.py --images ./captured_images --output ./analysis --threshold 30

Input images should follow the camera_server naming convention:
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
from datetime import datetime


# ── ROI Selection ─────────────────────────────────────────────────────────

class ROISelector:
    """Interactive ROI selection for the tip region."""

    def __init__(self):
        self.roi = None
        self.drawing = False
        self.start = None

    def select(self, image, window_name="Select Tip Region (drag, then press ENTER)"):
        """Let user draw a rectangle around the tips. Returns (x, y, w, h)."""
        self.roi = None
        self.drawing = False
        clone = image.copy()

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                temp = clone.copy()
                cv2.rectangle(temp, self.start, (x, y), (0, 255, 0), 2)
                cv2.imshow(window_name, temp)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                x1, y1 = self.start
                x2, y2 = x, y
                self.roi = (
                    min(x1, x2), min(y1, y2),
                    abs(x2 - x1), abs(y2 - y1),
                )
                cv2.rectangle(clone, (self.roi[0], self.roi[1]),
                              (self.roi[0] + self.roi[2], self.roi[1] + self.roi[3]),
                              (0, 255, 0), 2)
                cv2.imshow(window_name, clone)

        cv2.imshow(window_name, image)
        cv2.setMouseCallback(window_name, mouse_cb)

        print("Draw a rectangle around ALL 8 tips, then press ENTER.")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and self.roi is not None:  # Enter
                break
            elif key == 27:  # Escape
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()
        return self.roi

    def save(self, roi, path):
        """Save ROI to file for reuse."""
        with open(path, "w") as f:
            f.write(f"{roi[0]},{roi[1]},{roi[2]},{roi[3]}")
        print(f"[ROI] Saved to {path}")

    def load(self, path):
        """Load ROI from file."""
        with open(path, "r") as f:
            parts = f.read().strip().split(",")
            roi = tuple(int(p) for p in parts)
            print(f"[ROI] Loaded from {path}: {roi}")
            return roi


def crop_roi(image, roi):
    """Crop image to ROI (x, y, w, h)."""
    x, y, w, h = roi
    return image[y:y+h, x:x+w]


# ── Tip Segmentation ─────────────────────────────────────────────────────

def segment_tips(roi_image, num_tips=8):
    """
    Divide the ROI into individual tip regions.
    Assumes tips are arranged horizontally (8-channel in a row).
    Returns list of (x_start, x_end) column slices.
    """
    h, w = roi_image.shape[:2]
    tip_width = w // num_tips
    tips = []
    for i in range(num_tips):
        x_start = i * tip_width
        x_end = (i + 1) * tip_width if i < num_tips - 1 else w
        tips.append((x_start, x_end))
    return tips


def extract_tip_images(roi_image, tip_slices):
    """Extract individual tip images from ROI."""
    return [roi_image[:, x1:x2] for x1, x2 in tip_slices]


# ── Liquid Detection Methods ─────────────────────────────────────────────

def detect_by_subtraction(tip_img, baseline_tip_img, threshold=25):
    """
    Compare a tip image against the empty baseline.
    Returns: diff_ratio (0.0 = identical to empty, 1.0 = completely different)
    """
    gray = cv2.cvtColor(tip_img, cv2.COLOR_BGR2GRAY)
    base_gray = cv2.cvtColor(baseline_tip_img, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray, base_gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    total_pixels = mask.shape[0] * mask.shape[1]
    changed_pixels = np.count_nonzero(mask)

    return changed_pixels / total_pixels if total_pixels > 0 else 0.0, mask


def detect_meniscus(tip_img, baseline_tip_img, threshold=25):
    """
    Find the liquid meniscus (top of liquid) in a tip.
    Returns: liquid_height_ratio (0.0 = empty, 1.0 = full)
             meniscus_y (pixel row of meniscus, -1 if not found)
    """
    gray = cv2.cvtColor(tip_img, cv2.COLOR_BGR2GRAY)
    base_gray = cv2.cvtColor(baseline_tip_img, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray, base_gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find topmost and bottommost changed rows
    row_sums = np.sum(mask, axis=1)
    changed_rows = np.where(row_sums > 0)[0]

    if len(changed_rows) == 0:
        return 0.0, -1

    meniscus_y = changed_rows[0]  # Top of liquid
    bottom_y = changed_rows[-1]   # Bottom of liquid
    tip_height = mask.shape[0]

    liquid_height = bottom_y - meniscus_y
    liquid_height_ratio = liquid_height / tip_height if tip_height > 0 else 0.0

    return liquid_height_ratio, meniscus_y


def detect_by_color(tip_img, baseline_tip_img):
    """
    Detect liquid by color shift (works well for colored liquids).
    Compares average color in the tip region.
    Returns: color_diff (magnitude of color change)
    """
    avg_color = np.mean(tip_img, axis=(0, 1))
    avg_base = np.mean(baseline_tip_img, axis=(0, 1))
    color_diff = np.linalg.norm(avg_color - avg_base)
    return float(color_diff)


def detect_green_liquid(tip_img, hue_low=35, hue_high=85, sat_min=40, val_min=40):
    """
    Detect GREEN liquid directly using HSV color segmentation.
    No baseline needed — just looks for green pixels.

    HSV ranges for green:
      Hue: 35-85 (covers yellow-green to blue-green)
      Saturation: >40 (ignore grayish pixels)
      Value: >40 (ignore very dark pixels)

    Returns:
      green_ratio: fraction of tip that is green (0.0 = no green, 1.0 = all green)
      green_mask: binary mask of green pixels
      green_height_ratio: how far down the tip the green extends (liquid level)
      meniscus_y: pixel row of top of green region (-1 if none)
    """
    hsv = cv2.cvtColor(tip_img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([hue_low, sat_min, val_min])
    upper_green = np.array([hue_high, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    total_pixels = green_mask.shape[0] * green_mask.shape[1]
    green_pixels = np.count_nonzero(green_mask)
    green_ratio = green_pixels / total_pixels if total_pixels > 0 else 0.0

    # Find meniscus (top of green region)
    row_sums = np.sum(green_mask, axis=1)
    green_rows = np.where(row_sums > 0)[0]

    if len(green_rows) == 0:
        return green_ratio, green_mask, 0.0, -1

    meniscus_y = green_rows[0]
    bottom_y = green_rows[-1]
    tip_height = green_mask.shape[0]
    green_height_ratio = (bottom_y - meniscus_y) / tip_height if tip_height > 0 else 0.0

    return green_ratio, green_mask, green_height_ratio, meniscus_y


def detect_by_intensity(tip_img, baseline_tip_img):
    """
    Detect liquid by overall brightness change.
    Liquid in tips typically makes them darker or changes transparency.
    Returns: intensity_change (positive = brighter, negative = darker)
    """
    gray = cv2.cvtColor(tip_img, cv2.COLOR_BGR2GRAY).astype(float)
    base_gray = cv2.cvtColor(baseline_tip_img, cv2.COLOR_BGR2GRAY).astype(float)
    return float(np.mean(gray) - np.mean(base_gray))


# ── Image Categorization ─────────────────────────────────────────────────

def categorize_images(image_dir):
    """
    Sort images into categories based on camera_server naming convention.
    Returns dict: { category: [filepath, ...] }
    """
    categories = {
        "pick_up_tip": [],       # Empty tips baseline
        "before_aspirate": [],   # Should be empty
        "after_aspirate": [],    # Should have liquid
        "before_dispense": [],   # Should have liquid
        "after_dispense": [],    # Should be empty
        "drop_tip": [],          # After dropping
        "other": [],
    }

    files = sorted(os.listdir(image_dir))
    for f in files:
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        filepath = os.path.join(image_dir, f)
        matched = False
        for cat in categories:
            if cat in f.lower():
                categories[cat].append(filepath)
                matched = True
                break
        if not matched:
            categories["other"].append(filepath)

    for cat, files in categories.items():
        if files:
            print(f"  {cat}: {len(files)} images")

    return categories


def extract_column_from_filename(filepath):
    """Extract column info from filename (e.g., 'col_A3' -> 'A3')."""
    basename = os.path.basename(filepath)
    match = re.search(r'col_([A-H]\d+)', basename)
    return match.group(1) if match else "unknown"


# ── Main Analysis ─────────────────────────────────────────────────────────

def analyze_images(image_dir, output_dir, threshold=25, num_tips=8, roi_path=None):
    """Full analysis pipeline."""

    os.makedirs(output_dir, exist_ok=True)

    # 1. Categorize images
    print("\n[1] Categorizing images...")
    categories = categorize_images(image_dir)

    # 2. Get baseline (empty tips) from pick_up_tip images
    baseline_files = categories["pick_up_tip"] or categories["before_aspirate"]
    if not baseline_files:
        print("[ERROR] No baseline images found (need pick_up_tip or before_aspirate)")
        return

    baseline_img = cv2.imread(baseline_files[0])
    if baseline_img is None:
        print(f"[ERROR] Cannot read baseline image: {baseline_files[0]}")
        return
    print(f"\n[2] Using baseline: {os.path.basename(baseline_files[0])}")

    # 3. Select or load ROI
    roi_file = roi_path or os.path.join(output_dir, "roi.txt")
    selector = ROISelector()

    if os.path.exists(roi_file):
        roi = selector.load(roi_file)
    else:
        print("\n[3] Select the region containing the tips...")
        roi = selector.select(baseline_img)
        if roi is None:
            print("[ERROR] No ROI selected")
            return
        selector.save(roi, roi_file)

    # 4. Segment tips in baseline
    baseline_roi = crop_roi(baseline_img, roi)
    tip_slices = segment_tips(baseline_roi, num_tips)
    baseline_tips = extract_tip_images(baseline_roi, tip_slices)

    # 5. Analyze each image
    print(f"\n[4] Analyzing images (threshold={threshold})...")
    results = []

    analysis_categories = ["before_aspirate", "after_aspirate", "before_dispense", "after_dispense"]

    for cat in analysis_categories:
        for filepath in categories[cat]:
            img = cv2.imread(filepath)
            if img is None:
                continue

            img_roi = crop_roi(img, roi)
            test_tips = extract_tip_images(img_roi, tip_slices)
            column = extract_column_from_filename(filepath)

            row = {
                "filename": os.path.basename(filepath),
                "action": cat,
                "column": column,
            }

            # Analyze each tip
            tip_diffs = []
            tip_heights = []
            tip_colors = []
            tip_intensities = []
            tip_green_ratios = []
            tip_green_heights = []

            for t in range(num_tips):
                diff_ratio, mask = detect_by_subtraction(
                    test_tips[t], baseline_tips[t], threshold
                )
                height_ratio, meniscus_y = detect_meniscus(
                    test_tips[t], baseline_tips[t], threshold
                )
                color_diff = detect_by_color(test_tips[t], baseline_tips[t])
                intensity = detect_by_intensity(test_tips[t], baseline_tips[t])
                green_ratio, green_mask, green_height, green_meniscus = detect_green_liquid(
                    test_tips[t]
                )

                tip_diffs.append(diff_ratio)
                tip_heights.append(height_ratio)
                tip_colors.append(color_diff)
                tip_intensities.append(intensity)
                tip_green_ratios.append(green_ratio)
                tip_green_heights.append(green_height)

                row[f"tip_{t+1}_diff"] = f"{diff_ratio:.4f}"
                row[f"tip_{t+1}_height"] = f"{height_ratio:.4f}"
                row[f"tip_{t+1}_color"] = f"{color_diff:.2f}"
                row[f"tip_{t+1}_intensity"] = f"{intensity:.2f}"
                row[f"tip_{t+1}_green"] = f"{green_ratio:.4f}"
                row[f"tip_{t+1}_green_height"] = f"{green_height:.4f}"

            # Summary stats
            row["avg_diff"] = f"{np.mean(tip_diffs):.4f}"
            row["avg_height"] = f"{np.mean(tip_heights):.4f}"
            row["avg_color"] = f"{np.mean(tip_colors):.2f}"
            row["avg_intensity"] = f"{np.mean(tip_intensities):.2f}"
            row["avg_green"] = f"{np.mean(tip_green_ratios):.4f}"
            row["avg_green_height"] = f"{np.mean(tip_green_heights):.4f}"

            # Liquid detected? GREEN is the primary detector now
            avg_green = np.mean(tip_green_ratios)
            avg_diff = np.mean(tip_diffs)
            row["liquid_detected"] = "YES" if (avg_green > 0.02 or avg_diff > 0.05) else "NO"

            results.append(row)

            # Save annotated image
            annotated = img.copy()
            x, y, w, h = roi
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for t, (x1, x2) in enumerate(tip_slices):
                tx = x + x1
                green_pct = tip_green_ratios[t] * 100
                label = f"T{t+1}: {green_pct:.1f}%"
                color = (0, 0, 255) if tip_green_ratios[t] > 0.02 else (0, 255, 0)
                cv2.putText(annotated, label, (tx, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # Save green mask overlay for debugging
            green_overlay = img_roi.copy()
            for t, (x1, x2) in enumerate(tip_slices):
                _, g_mask, _, _ = detect_green_liquid(test_tips[t])
                green_overlay[0:g_mask.shape[0], x1:x1+g_mask.shape[1], 1] = cv2.add(
                    green_overlay[0:g_mask.shape[0], x1:x1+g_mask.shape[1], 1],
                    (g_mask * 0.5).astype(np.uint8)
                )

            ann_path = os.path.join(output_dir, f"annotated_{os.path.basename(filepath)}")
            cv2.imwrite(ann_path, annotated)
            mask_path = os.path.join(output_dir, f"green_mask_{os.path.basename(filepath)}")
            cv2.imwrite(mask_path, green_overlay)

            status = "🟢 LIQUID" if row["liquid_detected"] == "YES" else "⚪ empty"
            print(f"  {cat:20s} | col {column:4s} | green={row['avg_green']} | diff={row['avg_diff']} | {status}")

    # 6. Save CSV
    if results:
        csv_path = os.path.join(output_dir, "liquid_analysis.csv")
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[5] Results saved to: {csv_path}")

    # 7. Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for cat in analysis_categories:
        cat_results = [r for r in results if r["action"] == cat]
        if cat_results:
            liquid_count = sum(1 for r in cat_results if r["liquid_detected"] == "YES")
            total = len(cat_results)
            avg_green = np.mean([float(r["avg_green"]) for r in cat_results])
            avg_diff = np.mean([float(r["avg_diff"]) for r in cat_results])
            print(f"  {cat:20s}: {liquid_count}/{total} with liquid | avg_green={avg_green:.4f} | avg_diff={avg_diff:.4f}")

    expected = {
        "before_aspirate": "NO",   # Should be empty
        "after_aspirate": "YES",   # Should have liquid
        "before_dispense": "YES",  # Should have liquid
        "after_dispense": "NO",    # Should be empty
    }
    print("\nExpected vs Detected:")
    for cat in analysis_categories:
        cat_results = [r for r in results if r["action"] == cat]
        for r in cat_results:
            match = "✓" if r["liquid_detected"] == expected[cat] else "✗"
            print(f"  {match} {r['filename']}: expected={expected[cat]}, got={r['liquid_detected']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect liquid in pipette tips")
    parser.add_argument("--images", default="./captured_images",
                        help="Directory with captured images (default: ./captured_images)")
    parser.add_argument("--output", default="./analysis",
                        help="Output directory (default: ./analysis)")
    parser.add_argument("--threshold", type=int, default=25,
                        help="Pixel difference threshold (default: 25)")
    parser.add_argument("--tips", type=int, default=8,
                        help="Number of tips (default: 8)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Force re-selection of tip ROI")
    args = parser.parse_args()

    if args.calibrate:
        roi_file = os.path.join(args.output, "roi.txt")
        if os.path.exists(roi_file):
            os.remove(roi_file)

    analyze_images(args.images, args.output, args.threshold, args.tips)
