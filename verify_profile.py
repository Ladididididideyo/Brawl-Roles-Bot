#!/usr/bin/env python3
"""
verify_profile.py  <image_path> [expected_tag]

1. Runs weighted OpenCV template matching to confirm the screenshot is the
   user's OWN Brawl Stars profile (gear icons, colour picker, QR button).

2. If expected_tag is supplied, also OCRs the tag region using system Tesseract
   and fuzzy-matches against the supplied tag (edit distance ≤ 2 on a 8-10 char
   tag, to handle font misreads on the Nougat/LilitaOne typefaces).

Prints a single JSON object to stdout:
{
  "ownProfile":   bool,
  "confidence":   float,       # 0.0–1.0 weighted CV score
  "details":      { element: score },
  "tagVerified":  bool | null, # null if no tag supplied
  "tagOcr":       str  | null  # what OCR actually read
}
"""

import sys, json, os, subprocess, tempfile
import cv2
import numpy as np

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

# Brawl Stars tags only use these characters (Supercell excludes ambiguous ones)
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


def ocr_tag(image):
    """
    Crops the tag region, preprocesses with CLAHE contrast enhancement,
    tries multiple threshold values, and returns the longest valid tag read.
    Uses system Tesseract with the restricted Brawl Stars charset.
    """
    ih, iw = image.shape[:2]
    best_text = ''

    # Scan a vertical band where the tag always lives (28–35% down, left 30% wide)
    for frac in [0.285, 0.295, 0.305, 0.315, 0.325, 0.335]:
        y0 = int(ih * frac)
        y1 = y0 + max(40, int(ih * 0.040))
        crop = image[y0:y1, 0:int(iw * 0.30)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # CLAHE boosts local contrast — essential for the Nougat font on blue bg
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        for thresh_val in [120, 130, 140, 150]:
            _, thresh = cv2.threshold(enhanced, thresh_val, 255, cv2.THRESH_BINARY)
            up     = cv2.resize(thresh, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
            padded = cv2.copyMakeBorder(up, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=0)

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                tmp = f.name
            cv2.imwrite(tmp, padded)

            try:
                r = subprocess.run(
                    ['tesseract', tmp, 'stdout', '--psm', '7',
                     '-c', f'tessedit_char_whitelist={BRAWL_CHARS}'],
                    capture_output=True, text=True, timeout=10
                )
                text = r.stdout.strip().upper().replace(' ', '')
                # Keep the longest result that looks like a tag
                if text.startswith('#') and len(text) >= 5 and len(text) > len(best_text):
                    best_text = text
            except Exception:
                pass
            finally:
                try: os.unlink(tmp)
                except: pass

    return best_text


def normalise_tag(tag):
    """
    Normalise a tag for fuzzy comparison.
    Collapses characters that look identical in the Brawl Stars Nougat font:
      Q / O / 0  →  0
      I / 1      →  1  (rare in BS tags but OCR may produce them)
    Strips spaces, uppercases, keeps only # and alphanumerics.
    Also strips accidental double ## from OCR.
    """
    s = tag.upper().replace(' ', '')
    # Remove leading duplicate #
    while s.startswith('##'):
        s = s[1:]
    result = []
    for c in s:
        if c in ('Q', 'O'):
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
        details[name] = round(score, 4)
        weighted_score += weight * (1.0 if score >= threshold else 0.0)
        total_weight   += weight

    confidence  = (weighted_score / total_weight) if total_weight > 0 else 0.0
    own_profile = confidence >= OWN_PROFILE_THRESHOLD

    # ── OCR tag verification ──────────────────────────────────────────────────
    tag_verified = None
    tag_ocr      = None

    if expected_tag is not None:
        tag_ocr = ocr_tag(image)
        norm_ocr = normalise_tag(tag_ocr)
        norm_exp = normalise_tag(expected_tag)

        if norm_ocr and norm_exp:
            dist      = edit_distance(norm_ocr, norm_exp)
            max_dist  = max(2, len(norm_exp) // 5)   # allow ≤2 misreads (≤20%)
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