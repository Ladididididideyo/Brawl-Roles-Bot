#!/usr/bin/env python3
"""
verify_profile.py  <image_path>

Runs weighted template matching to determine whether a Brawl Stars screenshot
is of the user's OWN profile (which shows gear/settings icons, colour picker,
QR code button, and card gear icon) vs. someone else's profile.

Exits with JSON printed to stdout:
  { "ownProfile": bool, "confidence": float, "details": { <element>: score } }
"""

import sys, json, os
import cv2
import numpy as np

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')

# Each element: (filename, weight, threshold)
# threshold = minimum score to count as "present"
# weight    = how much it contributes to the confidence score
# Threshold is generous (0.75) — the own/other gap is huge (1.0 vs ≤0.55)
ELEMENTS = [
    ('gear_left.png',     0.30, 0.75),  # HIGH  — very distinctive, always present
    ('gear_right.png',    0.25, 0.75),  # HIGH  — very distinctive, always present
    ('colour_picker.png', 0.20, 0.75),  # HIGH  — distinctive rainbow circle
    ('qr_button.png',     0.15, 0.75),  # HIGH  — distinctive QR pattern
    ('card_gear.png',     0.07, 0.75),  # MED   — on character portrait
    ('plus_button.png',   0.03, 0.75),  # LOW   — may not be present if slots filled
]

# Weighted score >= this → own profile
OWN_PROFILE_THRESHOLD = 0.60  # conservative; trivially met at 1.0, missed at ≤0.55


def match_score(image: np.ndarray, template: np.ndarray) -> float:
    """Return the best TM_CCOEFF_NORMED score for template anywhere in image."""
    th, tw = template.shape[:2]
    ih, iw = image.shape[:2]
    if th > ih or tw > iw:
        return 0.0
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    return float(np.max(result))


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: verify_profile.py <image_path>"}))
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    # Fallback: read raw bytes and decode via numpy (handles edge-case formats)
    if image is None:
        raw = np.frombuffer(open(image_path, 'rb').read(), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    if image is None:
        print(json.dumps({"error": f"Could not decode image: {image_path}"}))
        sys.exit(1)

    details = {}
    weighted_score = 0.0
    total_weight   = 0.0

    for filename, weight, threshold in ELEMENTS:
        tmpl_path = os.path.join(TEMPLATE_DIR, filename)
        template  = cv2.imread(tmpl_path)
        if template is None:
            # Missing template — skip gracefully, redistribute weight
            continue

        score = match_score(image, template)
        name  = filename.replace('.png', '')
        details[name] = round(score, 4)

        present = score >= threshold
        weighted_score += weight * (1.0 if present else 0.0)
        total_weight   += weight

    # Normalise in case some templates were missing
    confidence = (weighted_score / total_weight) if total_weight > 0 else 0.0
    own_profile = confidence >= OWN_PROFILE_THRESHOLD

    print(json.dumps({
        "ownProfile":  own_profile,
        "confidence":  round(confidence, 4),
        "details":     details,
    }))


if __name__ == '__main__':
    main()