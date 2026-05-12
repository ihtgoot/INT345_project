# -*- coding: utf-8 -*-
"""
=========================================================
  Image Resizer — Normalize All Images to Same Size
=========================================================
PURPOSE:
  Scans through all images in 'clean/' and 'distorted/'
  folders (including all batch subfolders) and resizes
  every image to the same TARGET resolution.

HOW TO RUN:
  python resize_images.py

  OR with custom size:
  python resize_images.py --width 512 --height 512

  OR to preview only (no changes):
  python resize_images.py --dry-run
=========================================================
"""

import cv2
import os
import argparse
from tqdm import tqdm

# =========================================================
# CONFIGURATION — Change target size here
# =========================================================
TARGET_WIDTH  = 256   # <-- Set your desired width  (pixels)
TARGET_HEIGHT = 256   # <-- Set your desired height (pixels)

# Folders to process (relative to this script)
FOLDERS_TO_PROCESS = ["clean", "distorted"]

# Image extensions to look for
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
# =========================================================


def collect_all_images(base_folder):
    """
    Walks through a folder and all its subfolders,
    collecting paths to every image file found.
    """
    image_paths = []
    for root, dirs, files in os.walk(base_folder):
        dirs.sort()  # process batches in order
        for filename in sorted(files):
            if filename.lower().endswith(VALID_EXTENSIONS):
                image_paths.append(os.path.join(root, filename))
    return image_paths


def check_resolution(image_paths, sample_limit=20):
    """
    Checks the resolution of a sample of images
    and reports how many unique sizes exist.
    """
    sizes = {}
    sample = image_paths[:sample_limit]
    for path in sample:
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            key = f"{w}x{h}"
            sizes[key] = sizes.get(key, 0) + 1
    return sizes


def resize_all_images(image_paths, target_w, target_h, dry_run=False):
    """
    Resizes all images in the list to target_w x target_h.
    Overwrites the original files in-place.
    """
    already_correct = 0
    resized = 0
    failed = 0

    for path in tqdm(image_paths, desc="Resizing Images"):
        img = cv2.imread(path)

        if img is None:
            print(f"  [SKIP] Could not read: {path}")
            failed += 1
            continue

        h, w = img.shape[:2]

        # Skip if already correct size
        if w == target_w and h == target_h:
            already_correct += 1
            continue

        if not dry_run:
            # Resize using high-quality Lanczos interpolation
            # - INTER_LANCZOS4 is best for downscaling (sharper than INTER_AREA)
            # - INTER_CUBIC is best for upscaling
            if w > target_w or h > target_h:
                interpolation = cv2.INTER_LANCZOS4  # shrinking → Lanczos
            else:
                interpolation = cv2.INTER_CUBIC     # growing  → Cubic

            resized_img = cv2.resize(img, (target_w, target_h), interpolation=interpolation)
            cv2.imwrite(path, resized_img)

        resized += 1

    return resized, already_correct, failed


def main():
    parser = argparse.ArgumentParser(description="Resize all images to the same resolution.")
    parser.add_argument("--width",   type=int, default=TARGET_WIDTH,  help=f"Target width  (default: {TARGET_WIDTH})")
    parser.add_argument("--height",  type=int, default=TARGET_HEIGHT, help=f"Target height (default: {TARGET_HEIGHT})")
    parser.add_argument("--dry-run", action="store_true",             help="Preview only — do NOT modify any files")
    args = parser.parse_args()

    target_w = args.width
    target_h = args.height
    dry_run  = args.dry_run

    # Locate base directory (same folder as this script)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 55)
    print("  IMAGE RESIZER — Normalize All Images to Same Size")
    print("=" * 55)
    print(f"  Target Resolution : {target_w} x {target_h} pixels")
    print(f"  Mode              : {'DRY RUN (no files changed)' if dry_run else 'LIVE (files will be overwritten)'}")
    print("=" * 55)

    total_resized = 0
    total_skipped = 0
    total_failed  = 0

    for folder_name in FOLDERS_TO_PROCESS:
        folder_path = os.path.join(base_dir, folder_name)

        if not os.path.isdir(folder_path):
            print(f"\n[WARNING] Folder not found, skipping: {folder_path}")
            continue

        print(f"\n[FOLDER] Scanning: {folder_name}/")
        image_paths = collect_all_images(folder_path)
        print(f"   Found {len(image_paths)} images total")

        if len(image_paths) == 0:
            continue

        # Show sample of current resolutions
        print("   Sampling current resolutions...")
        sample_sizes = check_resolution(image_paths, sample_limit=30)
        print("   Detected sizes (from sample):")
        for size, count in sorted(sample_sizes.items()):
            marker = " (already correct)" if size == f"{target_w}x{target_h}" else ""
            print(f"     {size}  ->  {count} images{marker}")

        # Run resize
        print(f"   Resizing to {target_w}x{target_h}...")
        resized, skipped, failed = resize_all_images(image_paths, target_w, target_h, dry_run)
        total_resized += resized
        total_skipped += skipped
        total_failed  += failed

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  [OK]    Resized          : {total_resized} images")
    print(f"  [SKIP]  Already correct  : {total_skipped} images")
    print(f"  [FAIL]  Failed / skipped : {total_failed} images")
    if dry_run:
        print("\n  [!] DRY RUN -- No files were actually changed.")
        print("  Run without --dry-run to apply changes.")
    else:
        print(f"\n  All done! Images are now {target_w}x{target_h} pixels.")
    print("=" * 55)


if __name__ == "__main__":
    main()
