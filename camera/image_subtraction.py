"""
Image Subtraction Script

Input: a folder called "input" containing all of the images
Output: Image depicting difference, image of mask, csv with changed pixels

Usage:
    python image_subtraction.py
    python image_subtraction.py --input ./my_images --output ./my_results --threshold 30
"""

import cv2
import numpy as np
import os
import re
import csv
import argparse


def subtract_images(before, after, threshold=25):
    """
    Core subtraction logic (no file I/O here).
    """

    if before.shape != after.shape:
        raise ValueError("Images must have the same shape")

    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(before_gray, after_gray)

    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # clean up noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    changed_pixels = np.count_nonzero(mask)

    return diff, mask, changed_pixels


def extract_id(filename):
    """
    Extract numeric ID from filename (e.g., sample_001_before.jpg -> 001)
    Adjust regex if your naming scheme differs.
    """
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None


def process_directory(input_dir, output_dir, threshold=25):
    """
    Process all before/after image pairs in a directory.
    """

    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(input_dir)

    before_images = {}
    after_images = {}

    # Separate before/after images
    for f in files:
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_id = extract_id(f)
        if img_id is None:
            continue

        if "before" in f.lower():
            before_images[img_id] = f
        elif "after" in f.lower():
            after_images[img_id] = f

    results = []

    # Process pairs
    for img_id in before_images:
        if img_id not in after_images:
            print(f"Skipping ID {img_id}: missing pair")
            continue

        before_path = os.path.join(input_dir, before_images[img_id])
        after_path = os.path.join(input_dir, after_images[img_id])

        before = cv2.imread(before_path)
        after = cv2.imread(after_path)

        if before is None or after is None:
            print(f"Skipping ID {img_id}: failed to load")
            continue

        try:
            diff, mask, changed_pixels = subtract_images(
                before, after, threshold
            )
        except Exception as e:
            print(f"Skipping ID {img_id}: {e}")
            continue

        # Save outputs
        diff_path = os.path.join(output_dir, f"{img_id}_diff.jpg")
        mask_path = os.path.join(output_dir, f"{img_id}_mask.jpg")

        cv2.imwrite(diff_path, diff)
        cv2.imwrite(mask_path, mask)

        results.append({
            "id": img_id,
            "before_image": before_images[img_id],
            "after_image": after_images[img_id],
            "changed_pixels": changed_pixels
        })

        print(f"Processed ID {img_id}: {changed_pixels} pixels")

    # Save results to CSV
    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["id", "before_image", "after_image", "changed_pixels"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nDone. Results saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image subtraction for before/after pairs")
    parser.add_argument("--input", default="input", help="Input directory (default: input)")
    parser.add_argument("--output", default="output", help="Output directory (default: output)")
    parser.add_argument("--threshold", type=int, default=25, help="Difference threshold (default: 25)")
    args = parser.parse_args()

    process_directory(args.input, args.output, args.threshold)
