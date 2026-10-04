

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Author: Thanh Phu Phan
# Last Modified: 2026-10-01

"""
Task 2: localise the reading inside a rectified device image (output of Task 1).

  - BPM LCD     -> one crop per seven-segment digit, saved as d1.png, d2.png, ...
                   in reading order (systolic row, diastolic row, pulse row,
                   each left to right).
  - Thermometer -> one crop, t.png, covering the top of the liquid column and
                   the two numbered scale marks around it.

Everything here is classical OpenCV: the input is already upright and tightly
cropped, so no learned model is needed. All sizes are relative to the image
width so the code does not depend on the input resolution.
"""

import glob
import os

import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

# An upright thermometer plate is much taller than it is wide; a BPM LCD is not.
THERMO_MIN_ASPECT = 2.5


def save_output(output_path, content, output_type='txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_type == 'txt':
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"Text file saved at: {output_path}")
    elif output_type == 'image':
        # Assuming 'content' is a valid image object, e.g., from OpenCV
        cv2.imwrite(output_path, content)
        print(f"Image saved at: {output_path}")
    else:
        print("Unsupported output type. Use 'txt' or 'image'.")


# --------------------------------------------------------------------------- #
# LCD digits
# --------------------------------------------------------------------------- #

def lcd_segment_mask(img):
    """Binary mask (255 = dark LCD segment) with uneven lighting removed."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w = gray.shape[1]
    # Dividing by a heavily blurred copy flattens shadows and glare gradients.
    background = cv2.GaussianBlur(gray, (0, 0), w / 10)
    flat = cv2.divide(gray, background, scale=255)
    _, mask = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def find_lcd_digits(img):
    """Return digit boxes (x, y, w, h) in reading order."""
    mask = lcd_segment_mask(img)
    h, w = mask.shape

    # Segments of one digit are separated by thin gaps; digits by much wider
    # ones. A small dilation joins each digit's segments into one blob.
    k = max(3, int(round(w * 0.025)))
    joined = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(joined)

    candidates = []
    for i in range(1, num):
        x, y, bw, bh, _ = stats[i]
        # The LCD holds three rows of digits, so a real digit is well under
        # half the image height. Bezel edges and shadows along the border are
        # far taller (or wider, caught by the aspect test below). A size test
        # rather than "touches the border" keeps digits clipped by Task 1.
        if bh > 0.45 * h:
            continue
        # Tight box from the undilated segment pixels of this blob.
        ys, xs = np.nonzero((labels == i) & (mask > 0))
        if len(xs) == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        # Seven-segment digits are clearly taller than wide; icons like "OK"
        # or the heart are roughly square.
        if (y1 - y0) < 1.4 * (x1 - x0):
            continue
        candidates.append((x0, y0, x1 - x0, y1 - y0))

    if not candidates:
        return []

    # Drop small specks: the pulse digits are the smallest real ones (~60% of
    # the systolic height).
    max_h = max(c[3] for c in candidates)
    digits = [c for c in candidates if c[3] >= 0.4 * max_h]

    # Group into rows by vertical centre, then read top-to-bottom, left-to-right.
    digits.sort(key=lambda c: c[1] + c[3] / 2)
    rows = []
    for c in digits:
        cy = c[1] + c[3] / 2
        if rows and abs(cy - rows[-1][0]) < 0.5 * c[3]:
            rows[-1][1].append(c)
        else:
            rows.append([cy, [c]])
    return [c for _, row in rows for c in sorted(row, key=lambda c: c[0])]


def crop_box(img, box, pad_frac=0.01):
    """Crop (x, y, w, h) with a small padding, clipped to the image."""
    x, y, bw, bh = box
    pad = int(round(pad_frac * max(bw, bh)))
    h, w = img.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    return img[y0:y1, x0:x1]


def extract_lcd_digits(img):
    """List of digit crops in reading order."""
    return [crop_box(img, box) for box in find_lcd_digits(img)]


# --------------------------------------------------------------------------- #
# Thermometer
# --------------------------------------------------------------------------- #

def fit_liquid_line(response, max_tilt_deg=6.0):
    """
    Find the line x = a*y + b along which the liquid response is strongest.

    A brute-force search over small tilts (a Hough transform restricted to
    near-vertical lines), so a plate Task 1 left slightly rotated still works.
    """
    h, w = response.shape
    # Sum over row blocks first: much faster, and the line barely moves
    # sideways within one block.
    block = max(1, h // 200)
    n = h // block
    blocks = response[:n * block].reshape(n, block, w).mean(axis=1)
    ys = (np.arange(n) + 0.5) * block
    yc = h / 2

    x_lo, x_hi = int(0.2 * w), int(0.8 * w)
    offsets = np.arange(x_lo, x_hi)
    max_slope = np.tan(np.radians(max_tilt_deg))
    best = (-1.0, 0.0, float(offsets[0]))
    for a in np.linspace(-max_slope, max_slope, 49):
        cols = np.round(offsets[None, :] + a * (ys[:, None] - yc)).astype(int)
        valid = (cols >= 0) & (cols < w)
        vals = np.where(valid, blocks[np.arange(n)[:, None], np.clip(cols, 0, w - 1)], 0.0)
        # Liquid is a 2-4 px line, so sum a 3-column window around each offset.
        score = vals.sum(axis=0)
        score = np.convolve(score, np.ones(3), mode="same")
        i = int(np.argmax(score))
        if score[i] > best[0]:
            best = (score[i], a, float(offsets[i]))
    _, a, x_centre = best
    return a, x_centre - a * yc


def find_liquid_column(img):
    """Return (x, y_top) of the liquid column: its column at the top, and the row of its top."""
    h, w = img.shape[:2]
    # Chroma (max - min channel) rather than HSV saturation: saturation
    # explodes on noisy dark pixels, which would make the black scale print
    # look "coloured".
    sat = (img.max(axis=2).astype(np.int16) - img.min(axis=2)).astype(np.uint8)
    sat = cv2.GaussianBlur(sat, (0, 0), 1.0)

    # The liquid is a thin, coloured vertical line; the scale print is black
    # (no chroma). A horizontal white top-hat on chroma keeps only narrow
    # coloured structures, i.e. the liquid.
    k = max(5, int(round(w * 0.05)))
    response = cv2.morphologyEx(sat, cv2.MORPH_TOPHAT, np.ones((1, k), np.uint8)).astype(np.float32)
    # Average along the (near-vertical) line direction to suppress pixel noise
    # without widening the line.
    response = cv2.GaussianBlur(response, (1, 0), sigmaX=0.1, sigmaY=max(1.0, h * 0.004))

    # Sample the response along the fitted line, a few pixels either side.
    a, b = fit_liquid_line(response)
    half = max(2, int(round(w * 0.006)))
    rows = np.arange(h)
    centres = np.clip(np.round(a * rows + b).astype(int), half, w - half - 1)
    row_profile = np.array([response[y, c - half:c + half + 1].mean() for y, c in zip(rows, centres)])
    row_profile = cv2.GaussianBlur(row_profile.reshape(-1, 1), (0, 0), max(1, h * 0.003)).ravel()

    # Otsu on the profile values splits "empty tube / plate" from "liquid",
    # so a constant offset from noise or lighting does not matter.
    scaled = cv2.normalize(row_profile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Clip the bulb's very high values so they do not drag the threshold up.
    scaled = np.minimum(scaled, np.uint8(np.percentile(scaled, 95)))
    thr, _ = cv2.threshold(scaled.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    on = scaled > thr

    # Bridge short dropouts (glare, tick shadows) before looking for the run.
    gap = max(3, int(round(h * 0.01)))
    on = cv2.morphologyEx(on.astype(np.uint8).reshape(-1, 1), cv2.MORPH_CLOSE,
                          np.ones((gap, 1), np.uint8)).ravel() > 0

    best_len, best_top, run_start = 0, None, None
    for y, v in enumerate(np.append(on, False)):
        if v and run_start is None:
            run_start = y
        elif not v and run_start is not None:
            if y - run_start > best_len:
                best_len, best_top = y - run_start, run_start
            run_start = None
    if best_top is None:
        return None, None
    return int(round(a * best_top + b)), best_top


def find_scale_labels(img, liquid_x):
    """Vertical centres of the numbered labels on the left (degC) scale, top to bottom."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Flatten lighting gradients first, as for the LCD, then threshold.
    flat = cv2.divide(gray, cv2.GaussianBlur(gray, (0, 0), w / 10), scale=255)
    _, mask = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num, _, stats, cents = cv2.connectedComponentsWithStats(mask)

    # Label digits are compact, glyph-sized blobs left of the tube. Tick marks
    # are thin and wide, so the aspect test removes them.
    glyphs = []
    for i in range(1, num):
        x, y, bw, bh, _ = stats[i]
        if not (0.04 * w <= bh <= 0.15 * w):
            continue
        if bw > 1.2 * bh or x + bw > liquid_x - 0.08 * w:
            continue
        glyphs.append((cents[i][1], bh))
    if not glyphs:
        return []

    # Glyphs of one label ("-20") share a row; merge them.
    glyphs.sort()
    rows = []
    for cy, bh in glyphs:
        if rows and cy - rows[-1][-1][0] < 0.5 * bh:
            rows[-1].append((cy, bh))
        else:
            rows.append([(cy, bh)])
    centres = [float(np.mean([g[0] for g in r])) for r in rows]

    # The labels are evenly spaced. Anything at either end with an unusual gap
    # (e.g. the "degC" heading) is dropped.
    while len(centres) >= 3:
        gaps = np.diff(centres)
        med = np.median(gaps)
        if abs(gaps[0] - med) > 0.3 * med:
            centres.pop(0)
        elif abs(gaps[-1] - med) > 0.3 * med:
            centres.pop()
        else:
            break
    return centres


def find_thermo_reading_box(img):
    """Return (y0, y1, liquid_x, liquid_top) for the thermometer reading crop."""
    h, w = img.shape[:2]
    liquid_x, top = find_liquid_column(img)
    if top is None:
        raise RuntimeError("liquid column not found")

    labels = find_scale_labels(img, liquid_x)
    if len(labels) >= 2:
        spacing = float(np.median(np.diff(labels)))
        # The pair of labels bracketing the liquid top (clamped to the ends).
        idx = int(np.searchsorted(labels, top))
        idx = min(max(idx, 1), len(labels) - 1)
        above, below = labels[idx - 1], labels[idx]
        margin = 0.3 * spacing
        y0, y1 = above - margin, below + margin
    else:
        # No readable scale: fall back to a window centred on the liquid top.
        y0, y1 = top - 0.3 * w, top + 0.3 * w

    y0 = int(max(0, round(y0)))
    y1 = int(min(h, round(y1)))
    return y0, y1, liquid_x, top


def extract_thermo_reading(img):
    y0, y1, _, _ = find_thermo_reading_box(img)
    return img[y0:y1, :]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def is_thermometer(img):
    h, w = img.shape[:2]
    return h / w >= THERMO_MIN_ASPECT


def list_images(image_path):
    if os.path.isfile(image_path):
        return [image_path]
    return sorted(p for p in glob.glob(os.path.join(image_path, "*"))
                  if p.lower().endswith(IMAGE_EXTS))


def run_task2(image_path, config):
    for path in list_images(image_path):
        img = cv2.imread(path)
        if img is None:
            print(f"Could not read image: {path}")
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join("output", "task2", name)

        try:
            if is_thermometer(img):
                save_output(os.path.join(out_dir, "t.png"), extract_thermo_reading(img), output_type='image')
            else:
                crops = extract_lcd_digits(img)
                if not crops:
                    print(f"No digits found in {path}")
                for i, crop in enumerate(crops, start=1):
                    save_output(os.path.join(out_dir, f"d{i}.png"), crop, output_type='image')
        except Exception as e:
            # One bad image must not stop the rest of the batch.
            print(f"Task 2 failed on {path}: {e}")
