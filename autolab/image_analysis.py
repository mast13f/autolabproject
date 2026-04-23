"""
Image Analysis — Dye-Biased Subtraction Pipeline
=================================================
Replaces the previous YOLO-NAS + HSV analysis with a lighter OpenCV-only
pipeline that compares a reference image (pick_up_tip — empty tips) against
a target image (after_dispense) to quantify residual liquid.

Scoring
-------
Each column gets a pixel-count measuring how much the tip changed between
pick-up and after-dispense.  Fewer residual pixels → higher score (0-100).

Dependencies: cv2, numpy  (standard library: os, re, csv, datetime)
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Tuneable constants ────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 10        # Minimum per-pixel increase in blue-dominance between ref and target
ABS_BLUE_THRESHOLD = 6        # Target pixel must also be this much more blue than red (filters grey shaft-edge noise)
ROI_TOP_FRACTION = 0.55       # Top of ROI band — tip bottoms start roughly mid-frame
ROI_BOTTOM_FRACTION = 0.95    # Bottom of ROI band — catch hanging droplets below tips too
MIN_CONTOUR_AREA = 60         # Small specks (< this) are noise from lighting drift on the background
DYE_COLOR = "blue"            # "red", "blue", or "green" — dictates which opponent channel we sample
ALIGN_IMAGES = False          # Enable ECC sub-pixel alignment (rarely needed — static camera)
MAX_RESIDUAL_PIXELS = 30000   # Worst-case dye-pixel count (score = 0). ~20k is a fully-residue frame.


# ── Image alignment ──────────────────────────────────────────────────────────

def align_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Align *img2* to *img1* using ECC translation correction.

    Returns the warped img2, or the original if alignment fails.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    try:
        _, warp_matrix = cv2.findTransformECC(
            gray1, gray2, warp_matrix, cv2.MOTION_TRANSLATION, criteria,
        )
        h, w = img1.shape[:2]
        return cv2.warpAffine(
            img2, warp_matrix, (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
    except cv2.error:
        return img2


# ── Dye-specific residue detection ───────────────────────────────────────────

def _dye_opponent(img: np.ndarray) -> np.ndarray:
    """Per-pixel dye-opponent channel (positive => pixel looks like the dye).

    For blue dye this is B - R. Using the signed int16 difference keeps both
    positive and negative values so we can compute real gains later.
    """
    f = img.astype(np.int16)
    dye = DYE_COLOR.lower()
    if dye == "blue":
        return f[:, :, 0] - f[:, :, 2]            # B - R
    if dye == "red":
        return f[:, :, 2] - f[:, :, 0]            # R - B
    if dye == "green":
        return f[:, :, 1] - (f[:, :, 2] + f[:, :, 0]) // 2
    raise ValueError(f"Unknown DYE_COLOR '{DYE_COLOR}'. Choose 'red', 'blue', or 'green'.")


def subtract_images(
    img1: np.ndarray,
    img2: np.ndarray,
    threshold: int = DEFAULT_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Detect dye residue in *img2* using *img1* (reference / empty tips) as baseline.

    Returns ``(diff_visual, mask, dye_pixel_count)``. A pixel is counted as
    dye residue only when (a) it became notably more dye-coloured between
    the reference and target, AND (b) the target pixel is absolutely
    dye-coloured (screens out shaft-edge grey noise and lighting drift).
    """
    if img1 is None or img2 is None:
        raise ValueError("One or both images could not be loaded.")
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes do not match: {img1.shape} vs {img2.shape}")

    h, w = img1.shape[:2]

    if ALIGN_IMAGES:
        img2 = align_images(img1, img2)

    dye_ref = _dye_opponent(img1)
    dye_tgt = _dye_opponent(img2)
    gain    = dye_tgt - dye_ref                    # +ve where target is bluer than ref
    diff_visual = np.clip(gain, 0, 255).astype(np.uint8)

    m_gain = gain >= int(threshold)
    m_abs  = dye_tgt >= int(ABS_BLUE_THRESHOLD)

    roi_top = int(h * ROI_TOP_FRACTION)
    roi_bot = int(h * ROI_BOTTOM_FRACTION)
    band = np.zeros((h, w), dtype=bool)
    band[roi_top:roi_bot, :] = True

    raw = (m_gain & m_abs & band).astype(np.uint8) * 255

    # Clean up isolated specks, then close small gaps inside true residue blobs.
    k = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Drop any contour below MIN_CONTOUR_AREA (isolated noise).
    mask = np.zeros_like(cleaned)
    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) >= MIN_CONTOUR_AREA:
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)

    return diff_visual, mask, int(np.count_nonzero(mask))


# ── Filename parsing ─────────────────────────────────────────────────────────

def parse_filename(filename: str) -> Optional[dict]:
    """Parse a captured-image filename into structured metadata.

    Expected format (produced by ``camera/server.py``):
        ``YYYYMMDD_HHMMSS_microseconds_event_col_X#.jpg``

    Returns ``None`` if the filename does not match.
    """
    name, _ = os.path.splitext(filename)
    m = re.match(r"^(\d{8})_(\d{6})_(\d+?)_(.+)$", name)
    if not m:
        return None
    date_str, time_str, micro_str, remainder = m.groups()
    try:
        ts = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        ts = ts.replace(microsecond=int(micro_str[:6].ljust(6, "0")))
    except ValueError:
        return None
    col_m = re.search(r"_col_([A-Za-z]\d+)", remainder)
    if not col_m:
        return None
    event_map = {
        "pick_up_tip_col_":     "pick_up_tip",
        "before_aspirate_col_": "before_aspirate",
        "after_aspirate_col_":  "after_aspirate",
        "before_dispense_col_": "before_dispense",
        "after_dispense_col_":  "after_dispense",
    }
    event = next((v for k, v in event_map.items() if remainder.startswith(k)), None)
    if event is None:
        return None
    return {"filename": filename, "timestamp": ts, "column": col_m.group(1), "event": event}


def find_nearest_later(baseline: dict, candidates: list) -> Optional[dict]:
    """Return the candidate whose timestamp is closest-after *baseline*."""
    later = [c for c in candidates if c["timestamp"] >= baseline["timestamp"]]
    if not later:
        return None
    return min(later, key=lambda c: (c["timestamp"] - baseline["timestamp"]).total_seconds())


# ── Scoring ──────────────────────────────────────────────────────────────────

def compute_dispense_score(px_after_dispense: int) -> float:
    """Map residual pixel count to a 0-100 score.

    100 = perfectly clean dispense (0 residual pixels).
    0   = MAX_RESIDUAL_PIXELS or more changed pixels.
    """
    fraction = px_after_dispense / MAX_RESIDUAL_PIXELS
    return float(100.0 * (1.0 - np.clip(fraction, 0.0, 1.0)))


# ── Visual overlays ──────────────────────────────────────────────────────────

def save_overlay(reference_img: np.ndarray, mask: np.ndarray, output_path: str):
    """Save a diagnostic overlay highlighting detected residual pixels."""
    overlay = reference_img.copy()
    tint = reference_img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(tint, [cnt], -1, (0, 0, 200), thickness=cv2.FILLED)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.addWeighted(tint, 0.35, overlay, 0.65, 0, overlay)
    cv2.imwrite(output_path, overlay)


# ── Batch measurement ────────────────────────────────────────────────────────

def measure_from_images(
    input_dir: str,
    output_dir: str,
    threshold: int = DEFAULT_THRESHOLD,
    save_visuals: bool = True,
) -> Dict[str, dict]:
    """Process a folder of captured images and return per-column results.

    Compares each column's ``pick_up_tip`` (reference / empty tips) against
    its ``after_dispense`` image to measure residual liquid.

    Returns a dict keyed by column name, e.g.::

        {"A1": {"px_after_dispense": 1234, "score": 75.3}, ...}
    """
    os.makedirs(output_dir, exist_ok=True)
    if save_visuals:
        for sub in ("diffs", "masks", "overlays"):
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    all_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    parsed: List[dict] = []
    for f in all_files:
        p = parse_filename(f)
        if p:
            parsed.append(p)
        else:
            print(f"  [SKIP] {f}")

    by_col: Dict[str, list] = {}
    for item in parsed:
        by_col.setdefault(item["column"], []).append(item)

    column_results: Dict[str, dict] = {}
    csv_rows: List[dict] = []

    for column, items in by_col.items():
        items.sort(key=lambda x: x["timestamp"])
        pickups = [x for x in items if x["event"] == "pick_up_tip"]
        if not pickups:
            print(f"  [WARN] No pick_up_tip for column {column}")
            continue

        for pickup in pickups:
            pickup_img = cv2.imread(os.path.join(input_dir, pickup["filename"]))
            if pickup_img is None:
                continue

            # Compare pick_up_tip vs after_dispense
            candidates = [x for x in items if x["event"] == "after_dispense"]
            match = find_nearest_later(pickup, candidates)
            if match is None:
                print(f"  [WARN] No after_dispense found after {pickup['filename']}")
                column_results[column] = {"px_after_dispense": None, "score": None}
                continue

            target_img = cv2.imread(os.path.join(input_dir, match["filename"]))
            if target_img is None:
                column_results[column] = {"px_after_dispense": None, "score": None}
                continue

            diff, mask, n_px = subtract_images(pickup_img, target_img, threshold)
            print(f"  {column} pick_up_tip vs after_dispense: {n_px} px")

            if save_visuals:
                lbl = (
                    f"{column}_{pickup['timestamp'].strftime('%Y%m%d_%H%M%S_%f')}"
                    f"_vs_after_dispense_{match['timestamp'].strftime('%Y%m%d_%H%M%S_%f')}"
                )
                cv2.imwrite(os.path.join(output_dir, "diffs", f"{lbl}_diff.jpg"), diff)
                cv2.imwrite(os.path.join(output_dir, "masks", f"{lbl}_mask.jpg"), mask)
                save_overlay(
                    target_img, mask,
                    os.path.join(output_dir, "overlays", f"{lbl}_overlay.jpg"),
                )

            csv_rows.append({
                "column": column,
                "comparison": "pick_up_tip_vs_after_dispense",
                "pickup_image": pickup["filename"],
                "target_image": match["filename"],
                "pickup_timestamp": pickup["timestamp"].isoformat(),
                "target_timestamp": match["timestamp"].isoformat(),
                "time_difference_seconds": (
                    match["timestamp"] - pickup["timestamp"]
                ).total_seconds(),
                "changed_pixels": n_px,
            })

            score = compute_dispense_score(n_px)
            column_results[column] = {"px_after_dispense": n_px, "score": score}

    # Write per-comparison CSV
    if csv_rows:
        csv_path = os.path.join(output_dir, "image_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    return column_results


def aggregate_score(column_results: Dict[str, dict]) -> Optional[float]:
    """Compute an overall score from per-column results.

    Returns the mean of valid column scores, minus a 20-point penalty
    per column that could not be scored (missing images, etc.).
    Returns ``None`` if no columns could be scored at all.
    """
    scores = [v["score"] for v in column_results.values()]
    valid = [s for s in scores if s is not None]
    if not valid:
        return None
    penalty = (len(scores) - len(valid)) / len(scores) * 20
    return max(0.0, float(np.mean(valid)) - penalty)
