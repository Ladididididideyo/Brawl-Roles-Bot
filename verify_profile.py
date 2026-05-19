#!/usr/bin/env python3

"""
verify_profile.py <image_path> [expected_tag]

1. Runs weighted OpenCV template matching to confirm the screenshot is the
   user's OWN Brawl Stars profile (gear icons, colour picker, QR button).
2. If expected_tag is supplied, also OCRs the tag region using system Tesseract
   and fuzzy-matches against the supplied tag (edit distance ≤ 2 on a 8-10 char
   tag, to handle font misreads on the Nougat/LilitaOne typefaces).

Prints a single JSON object to stdout:
{
  "ownProfile":  bool,
  "confidence":  float,       # 0.0–1.0 weighted CV score
  "details":     { element: score },
  "tagVerified": bool | null, # null if no tag supplied
  "tagOcr":      str | null   # what OCR actually read
}
"""

import sys, json, os, subprocess, tempfile
from collections import Counter
import cv2
import numpy as np

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

# Brawl Stars tags only use these 16 characters (Supercell excludes ambiguous ones)
BRAWL_CHARS = '#02389CGJLPQRUVY'

ELEMENTS = [
    ('gear_left.png',     0.30, 0.75),
    ('gear_right.png',    0.25, 0.75),
    ('colour_picker.png', 0.20, 0.75),
    ('qr_button.png',     0.15, 0.75),
    ('card_gear.png',     0.07, 0.75),
    ('plus_button.png',   0.03, 0.75),
]

OWN_PROFILE_THRESHOLD = 0.60


def match_score(image, template):
    th, tw = template.shape[:2]
    ih, iw = image.shape[:2]
    if th > ih or tw > iw:
        return 0.0
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    return float(np.max(result))


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def _run_tesseract(img_gray):
    """Upscale, pad, and OCR a grayscale image. Returns raw text or ''."""
    up = cv2.resize(img_gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    pad = cv2.copyMakeBorder(up, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=0)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        tmp = f.name
    cv2.imwrite(tmp, pad)
    try:
        r = subprocess.run(
            ['tesseract', tmp, 'stdout', '--psm', '7',
             '-c', f'tessedit_char_whitelist={BRAWL_CHARS}'],
            capture_output=True, text=True, timeout=10
        )
        return r.stdout.strip().upper().replace(' ', '')
    except Exception:
        return ''
    finally:
        try: os.unlink(tmp)
        except: pass


def _pick_best_candidate(candidates, target_len):
    """
    Given a list of raw OCR strings (possibly with noise), normalise them,
    filter to those within +-1 of target_len, then return the medoid:
    the candidate with the highest score of:
        score = frequency / (1 + mean_edit_distance_to_all_others)
    This rewards both consistency across passes and consensus with neighbours.
    """
    if not candidates:
        return ''
    norm = [normalise_tag(t) for t in candidates]
    filtered = [t for t in norm if abs(len(t) - target_len) <= 1 and t.startswith('#')]
    if not filtered:
        filtered = norm

    counts = Counter(filtered)
    unique = list(counts.keys())
    if len(unique) == 1:
        return unique[0]

    def score(t):
        freq = counts[t]
        dists = [edit_distance(t, other) for other in filtered if other != t]
        mean_d = sum(dists) / len(dists) if dists else 0
        return freq / (1 + mean_d)

    return max(unique, key=score)


def ocr_tag(image):
    """
    Crop the player-tag region, run multiple preprocessing passes, and
    return the best read using weighted-medoid consensus selection.

    Layout (confirmed on 2388x1080 screenshots, scales by fraction):
    - The tag (#XXXXXXXXX) sits immediately below the skull/avatar frame
      in the top-left of the profile screen.
    - Vertical:   y = 29.0% to 34.5% of image height
    - Horizontal: x = 2.0%  to 22.5% of image width
    - Text is light-grey/blue on dark-blue background (Nougat font with
      a dark outline), so both polarities of threshold + HSV masking are tried.

    Fixes vs original:
    1. Crop region moved to the correct position (was scanning 28-34%, missing
       the tag entirely — it lives at 29-34% but centred differently).
    2. Replaced "pick longest" with weighted-medoid consensus across all passes.
    3. Added HSV colour masking to isolate the light tag text.
    4. Added Otsu auto-threshold passes.
    """
    ih, iw = image.shape[:2]

    y0 = int(ih * 0.290)
    y1 = int(ih * 0.345)
    x0 = int(iw * 0.020)
    x1 = int(iw * 0.225)
    crop = image[y0:y1, x0:x1]

    gray     = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # HSV masks isolate the light-blue/white tag text
    hsv        = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_light = cv2.inRange(hsv,  np.array([95,   0, 140]), np.array([130, 120, 255]))
    mask_white = cv2.inRange(crop, np.array([150, 150, 150]), np.array([255, 255, 255]))
    mask_comb  = cv2.bitwise_or(mask_light, mask_white)

    all_texts = []

    def collect(img_gray):
        text = _run_tesseract(img_gray)
        if text.startswith('#') and len(text) >= 5:
            all_texts.append(text)

    # Threshold sweeps — both polarities
    for tv in [90, 100, 110, 120, 130, 140, 150]:
        _, th = cv2.threshold(enhanced, tv, 255, cv2.THRESH_BINARY)
        collect(th)
    for tv in [70, 80, 90, 100, 110, 120, 130]:
        _, th = cv2.threshold(enhanced, tv, 255, cv2.THRESH_BINARY_INV)
        collect(th)

    # Colour masks
    for mask in [mask_comb, mask_light, mask_white]:
        collect(mask)

    # Otsu (auto threshold, both polarities)
    _, otsu     = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    collect(otsu)
    collect(otsu_inv)

    # Brawl Stars tags are 9-10 characters including '#'
    TARGET_LEN = 10
    return _pick_best_candidate(all_texts, TARGET_LEN)


def normalise_tag(tag):
    """
    Normalise a tag for fuzzy comparison.

    Maps only characters that look identical to valid BS chars but are NOT
    themselves valid BS tag characters:
      O (letter) -> 0 (zero)  [O is not in the BS charset; 0 is]
      I (letter) -> 1          [not valid; OCR may produce it]

    IMPORTANT FIX: Q IS a valid Brawl Stars character and must NOT be mapped
    to 0. The original code mapped both Q and O to 0, which corrupted tags
    containing Q (e.g. #2V8PRQGQCU -> #2V8PR0G0CU, causing every match to fail).
    """
    s = tag.upper().replace(' ', '')
    while s.startswith('##'):
        s = s[1:]
    result = []
    for c in s:
        if c == 'O':
            result.append('0')
        elif c == 'I':
            result.append('1')
        elif c in BRAWL_CHARS or c.isalnum():
            result.append(c)
        elif c == '#' and not result:
            result.append(c)
    return ''.join(result)


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: verify_profile.py <image_path> [tag]"}))
        sys.exit(1)

    image_path   = sys.argv[1]
    expected_tag = sys.argv[2] if len(sys.argv) >= 3 else None

    # ── Load image ────────────────────────────────────────────────────────────
    image = cv2.imread(image_path)
    if image is None:
        raw   = np.frombuffer(open(image_path, 'rb').read(), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        print(json.dumps({"error": f"Could not decode image: {image_path}"}))
        sys.exit(1)

    # ── CV own-profile detection ──────────────────────────────────────────────
    details        = {}
    weighted_score = 0.0
    total_weight   = 0.0

    for filename, weight, threshold in ELEMENTS:
        tmpl_path = os.path.join(TEMPLATE_DIR, filename)
        template  = cv2.imread(tmpl_path)
        if template is None:
            continue
        score = match_score(image, template)
        name  = filename.replace('.png', '')
        details[name]   = round(score, 4)
        weighted_score += weight * (1.0 if score >= threshold else 0.0)
        total_weight   += weight

    confidence  = (weighted_score / total_weight) if total_weight > 0 else 0.0
    own_profile = confidence >= OWN_PROFILE_THRESHOLD

    # ── OCR tag verification ──────────────────────────────────────────────────
    tag_verified = None
    tag_ocr      = None

    if expected_tag is not None:
        tag_ocr  = ocr_tag(image)
        norm_ocr = normalise_tag(tag_ocr)
        norm_exp = normalise_tag(expected_tag)

        if norm_ocr and norm_exp:
            dist         = edit_distance(norm_ocr, norm_exp)
            max_dist     = max(2, len(norm_exp) // 5)
            tag_verified = dist <= max_dist
        else:
            tag_verified = False

    print(json.dumps({
        "ownProfile":  own_profile,
        "confidence":  round(confidence, 4),
        "details":     details,
        "tagVerified": tag_verified,
        "tagOcr":      tag_ocr,
    }))


if __name__ == '__main__':
    main()