#!/usr/bin/env python3
"""
verify_profile.py <image_path> [expected_tag]

1. Runs weighted OpenCV template matching to confirm the screenshot is the
   user's OWN Brawl Stars profile (gear icons, colour picker, QR button).

2. If expected_tag is supplied, OCRs the tag region using Tesseract and
   fuzzy-matches against the supplied tag.

   Strategy: collect ALL candidates containing '#' across every crop/thresh
   combination, then pick the one with the LOWEST edit distance to the expected
   tag rather than the longest string. This handles lower-resolution screenshots
   where no single pass gives a perfect read.

Prints a single JSON object to stdout:
{
  "ownProfile": bool,
  "confidence": float,
  "details": { element: score },
  "tagVerified": bool | null,
  "tagOcr": str | null
}
"""

import sys, json, os, subprocess, tempfile
import cv2
import numpy as np

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

# Brawl Stars tags only use these characters — no letter O, no I
BRAWL_CHARS = '#023456789CGJLPQRUVY'

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


def normalise_tag(tag):
    """
    O → 0 and I → 1 since those letters are never valid BS tag chars.
    Q is left as-is since it IS a valid BS tag character.
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


def ocr_tag(image, expected_tag=None):
    """
    Collects all '#'-containing OCR candidates across every crop/thresh combo,
    then returns whichever has the lowest edit distance to expected_tag.
    If no expected_tag, returns the longest candidate.

    Pixel-verified: tag bar sits at 26-35% down the full image, left 50% wide.
    """
    ih, iw = image.shape[:2]
    candidates = []  # list of normalised strings containing '#'

    scan_fracs = [0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35]

    for frac in scan_fracs:
        y0 = int(ih * frac)
        y1 = y0 + int(ih * 0.10)
        crop = image[y0:y1, 0:int(iw * 0.50)]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        for thresh_val in [80, 100, 120, 140]:
            for inv in [False, True]:
                mode = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
                _, thresh = cv2.threshold(enhanced, thresh_val, 255, mode)
                up = cv2.resize(thresh, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
                pad_val = 255 if inv else 0
                padded = cv2.copyMakeBorder(up, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=pad_val)

                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    tmp = f.name
                cv2.imwrite(tmp, padded)

                try:
                    r = subprocess.run(
                        ['tesseract', tmp, 'stdout', '--psm', '7',
                         '-c', f'tessedit_char_whitelist={BRAWL_CHARS}'],
                        capture_output=True, text=True, timeout=10
                    )
                    text = r.stdout.strip().upper().replace(' ', '').replace('\n', '')
                    print(f'[ocr_tag] frac={frac} thresh={thresh_val} inv={inv} -> {repr(text)}', file=sys.stderr)

                    if '#' in text:
                        candidate = normalise_tag(text[text.index('#'):])
                        if candidate.startswith('#') and len(candidate) >= 5:
                            candidates.append(candidate)

                except Exception as e:
                    print(f'[ocr_tag] frac={frac} thresh={thresh_val} inv={inv} ERROR={e}', file=sys.stderr)
                finally:
                    try: os.unlink(tmp)
                    except: pass

    if not candidates:
        print('[ocr_tag] no candidates found', file=sys.stderr)
        return ''

    if expected_tag:
        norm_exp = normalise_tag(expected_tag)
        # Pick the candidate closest to the expected tag
        best = min(candidates, key=lambda c: edit_distance(c, norm_exp))
    else:
        # No expected tag — return the longest candidate
        best = max(candidates, key=len)

    print(f'[ocr_tag] candidates={candidates}', file=sys.stderr)
    print(f'[ocr_tag] best={repr(best)}', file=sys.stderr)
    return best


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: verify_profile.py <image_path> [tag]"}))
        sys.exit(1)

    image_path = sys.argv[1]
    expected_tag = sys.argv[2] if len(sys.argv) >= 3 else None

    # ── Load image ────────────────────────────────────────────────────────────
    image = cv2.imread(image_path)
    if image is None:
        raw = np.frombuffer(open(image_path, 'rb').read(), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        print(json.dumps({"error": f"Could not decode image: {image_path}"}))
        sys.exit(1)

    # ── CV own-profile detection ──────────────────────────────────────────────
    details = {}
    weighted_score = 0.0
    total_weight = 0.0

    for filename, weight, threshold in ELEMENTS:
        tmpl_path = os.path.join(TEMPLATE_DIR, filename)
        template = cv2.imread(tmpl_path)
        if template is None:
            continue
        score = match_score(image, template)
        name = filename.replace('.png', '')
        details[name] = round(score, 4)
        weighted_score += weight * (1.0 if score >= threshold else 0.0)
        total_weight += weight

    confidence = (weighted_score / total_weight) if total_weight > 0 else 0.0
    own_profile = confidence >= OWN_PROFILE_THRESHOLD

    # ── OCR tag verification ──────────────────────────────────────────────────
    tag_verified = None
    tag_ocr = None

    if expected_tag is not None:
        tag_ocr = ocr_tag(image, expected_tag)
        norm_ocr = normalise_tag(tag_ocr)
        norm_exp = normalise_tag(expected_tag)

        print(f'[tag] ocr={repr(norm_ocr)} expected={repr(norm_exp)}', file=sys.stderr)

        if norm_ocr and norm_exp:
            dist = edit_distance(norm_ocr, norm_exp)
            # Allow up to 1/3 of the tag length as misreads — handles low-res
            # screenshots while still being strict enough to prevent spoofing.
            # The CV own-profile check is the primary anti-spoofing gate.
            max_dist = max(3, len(norm_exp) // 3)
            print(f'[tag] edit_distance={dist} max_allowed={max_dist}', file=sys.stderr)
            tag_verified = dist <= max_dist
        else:
            tag_verified = False

    print(json.dumps({
        "ownProfile": own_profile,
        "confidence": round(confidence, 4),
        "details": details,
        "tagVerified": tag_verified,
        "tagOcr": tag_ocr,
    }))


if __name__ == '__main__':
    main()