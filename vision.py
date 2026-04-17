"""
AI Grand Prix — Vision Stack | Week 1 Deliverable
Gate Detection Foundations: HSV Filtering + Contour Detection

Takes an image and outputs the bounding box and center of a colored rectangle.
This is the foundation for gate detection in VQ1 where gates are visually
distinctive and highlighted.

Usage:
    python vision.py <image_path> [--color red|green|blue|yellow|orange]
    python vision.py --demo   # Generates a test image and runs detection

Requirements:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import argparse
import sys


# ---------------------------------------------------------------------------
# HSV colour ranges for common gate colours
# Each entry: list of (lower_hsv, upper_hsv) — some colours wrap around the
# hue wheel and need two ranges (e.g. red).
# ---------------------------------------------------------------------------
HSV_RANGES = {
    "red": [
        (np.array([0, 100, 100]),   np.array([10, 255, 255])),
        (np.array([160, 100, 100]), np.array([180, 255, 255])),
    ],
    "green": [
        (np.array([35, 80, 80]),  np.array([85, 255, 255])),
    ],
    "blue": [
        (np.array([90, 80, 80]),  np.array([130, 255, 255])),
    ],
    "yellow": [
        (np.array([20, 100, 100]), np.array([35, 255, 255])),
    ],
    "orange": [
        (np.array([10, 100, 100]), np.array([20, 255, 255])),
    ],
}


def create_hsv_mask(hsv_image: np.ndarray, color: str) -> np.ndarray:
    """Create a binary mask for the given colour from an HSV image."""
    if color not in HSV_RANGES:
        raise ValueError(f"Unknown color '{color}'. Choose from: {list(HSV_RANGES.keys())}")

    ranges = HSV_RANGES[color]
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_image, lower, upper))
    return mask


def detect_rectangle(image: np.ndarray, color: str = "red",
                     min_area_ratio: float = 0.001,
                     max_area_ratio: float = 0.9) -> dict | None:
    """
    Detect the largest brightly-coloured rectangle in an image.

    Parameters
    ----------
    image : np.ndarray
        BGR image (as loaded by cv2.imread).
    color : str
        Target colour name (key into HSV_RANGES).
    min_area_ratio : float
        Minimum contour area as a fraction of total image area.
    max_area_ratio : float
        Maximum contour area as a fraction of total image area.

    Returns
    -------
    dict or None
        {
            "bounding_box": (x, y, w, h),      # top-left corner + size in pixels
            "center": (cx, cy),                 # centre in pixel coordinates
            "area": int,                        # bounding box area in pixels
            "contour": np.ndarray,              # raw contour points
            "approx_poly": np.ndarray,          # simplified polygon vertices
        }
        Returns None if no suitable rectangle is found.
    """
    if image is None:
        raise ValueError("Image is None — check the file path.")

    h, w = image.shape[:2]
    total_area = h * w

    # --- Step 1: Convert BGR → HSV -------------------------------------------
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- Step 2: Build colour mask -------------------------------------------
    mask = create_hsv_mask(hsv, color)

    # --- Step 3: Clean up the mask with morphology ---------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- Step 4: Find contours -----------------------------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # --- Step 5: Filter and score contours -----------------------------------
    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Area filter
        if area < min_area_ratio * total_area or area > max_area_ratio * total_area:
            continue

        # Approximate polygon — check if roughly rectangular (4-6 vertices)
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        if 4 <= len(approx) <= 6 and area > best_area:
            best_area = area
            best = (cnt, approx)

    if best is None:
        # Fall back to largest contour even if not perfectly rectangular
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * total_area:
            return None
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        best = (cnt, approx)

    cnt, approx = best
    x, y, bw, bh = cv2.boundingRect(cnt)
    cx = x + bw // 2
    cy = y + bh // 2

    return {
        "bounding_box": (x, y, bw, bh),
        "center": (cx, cy),
        "area": bw * bh,
        "contour": cnt,
        "approx_poly": approx,
    }


def annotate_image(image: np.ndarray, detection: dict) -> np.ndarray:
    """Draw bounding box and centre marker on a copy of the image."""
    vis = image.copy()
    x, y, w, h = detection["bounding_box"]
    cx, cy = detection["center"]

    # Bounding box
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Centre crosshair
    size = 15
    cv2.line(vis, (cx - size, cy), (cx + size, cy), (0, 0, 255), 2)
    cv2.line(vis, (cx, cy - size), (cx, cy + size), (0, 0, 255), 2)

    # Label
    label = f"Center: ({cx}, {cy})  Size: {w}x{h}"
    cv2.putText(vis, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return vis


def generate_demo_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a synthetic scene with a coloured rectangular 'gate'."""
    # Sky-like gradient background
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        ratio = row / height
        img[row, :] = [int(180 - 80 * ratio), int(130 - 40 * ratio), int(50 + 30 * ratio)]  # BGR

    # Draw a red rectangle as a mock gate
    gate_x, gate_y = width // 3, height // 4
    gate_w, gate_h = width // 3, height // 2
    cv2.rectangle(img, (gate_x, gate_y), (gate_x + gate_w, gate_y + gate_h), (0, 0, 220), -1)
    # Inner cutout to simulate a gate opening
    inset = 20
    cv2.rectangle(
        img,
        (gate_x + inset, gate_y + inset),
        (gate_x + gate_w - inset, gate_y + gate_h - inset),
        (40, 40, 40),
        -1,
    )

    # Add a little noise for realism
    noise = np.random.randint(0, 15, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    return img


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Week 1 — Detect a coloured rectangle (gate proxy) in an image."
    )
    parser.add_argument("image", nargs="?", help="Path to input image")
    parser.add_argument("--color", default="red",
                        choices=list(HSV_RANGES.keys()),
                        help="Target gate colour (default: red)")
    parser.add_argument("--demo", action="store_true",
                        help="Generate a synthetic test image and run detection")
    parser.add_argument("--save", help="Path to save the annotated output image")
    args = parser.parse_args()

    # Load or generate image
    if args.demo:
        print("[demo] Generating synthetic gate image …", flush=True)
        image = generate_demo_image()
    elif args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: could not read image at '{args.image}'", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"Image size: {w}×{h}", flush=True)
    print(f"Target colour: {args.color}", flush=True)

    # Run detection
    result = detect_rectangle(image, color=args.color)

    if result is None:
        print("No rectangle detected.", flush=True)
        sys.exit(0)

    # Report
    x, y, bw, bh = result["bounding_box"]
    cx, cy = result["center"]
    print(f"\n{'='*40}", flush=True)
    print(f"  Bounding Box : x={x}, y={y}, w={bw}, h={bh}", flush=True)
    print(f"  Center       : ({cx}, {cy})", flush=True)
    print(f"  Area (px)    : {result['area']}", flush=True)
    print(f"  Vertices     : {len(result['approx_poly'])}", flush=True)
    print(f"{'='*40}\n", flush=True)

    # Save annotated image
    if args.save or args.demo:
        annotated = annotate_image(image, result)
        out_path = args.save or "demo_detection_output.png"
        cv2.imwrite(out_path, annotated)
        print(f"Annotated image saved to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
