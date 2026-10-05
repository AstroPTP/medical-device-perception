

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
# Last Modified: 2026-10-05

"""
Task 2: segment the reading inside a rectified device image (output of Task 1).

  - BPM LCD     -> one crop per seven-segment digit, saved as d1.png, d2.png, ...
                   in reading order (systolic row, diastolic row, pulse row,
                   each left to right).
  - Thermometer -> one crop, t.png, centred on the top of the liquid column and
                   reaching past the numbered scale labels above and below it.

Everything here is classical OpenCV: the input is already upright and tightly
cropped, so no learned model is needed. All sizes are relative to the image
size so the code does not depend on the input resolution.

Set "task2_debug: 1" in config.txt to also save output/task2/debug/<name>.png,
showing the detected digit boxes, or the liquid top, scale labels and ticks.
"""

import glob
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output", "task2")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

DEFAULT_DEBUG = 0      # 1 = also save a debug image per input

# An upright thermometer plate is much taller than it is wide; a BPM LCD is not.
# Only used when the file name does not say which device it is.
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


def dark_print_mask(img):
    """Binary mask (255 = dark print or LCD segment) with uneven lighting removed."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w = gray.shape[1]
    # Dividing by a heavily blurred copy flattens shadows and glare gradients.
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=w / 10)
    flat = cv2.divide(gray, background, scale=255)

    # Gaussian blur + Otsu's thresholding, then invert so the dark print is
    # the white foreground for connected component analysis (as in prac04).
    blur = cv2.GaussianBlur(flat, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(th)


# --------------------------------------------------------------------------- #
# LCD digits
# --------------------------------------------------------------------------- #

def lcd_segment_mask(img):
    """Mask of the dark LCD segments, with single-pixel noise removed."""
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(dark_print_mask(img), cv2.MORPH_OPEN, element)


def merge_fragments(boxes, max_h, shape):
    """
    Join boxes that are pieces of one digit.

    At low resolution the gap between a digit's upper and lower segments can
    be wider than the dilation bridges, so a "1" or "7" falls apart into a top
    and a bottom piece. Pieces of one digit share their columns, and together
    are no taller than a whole digit; a digit and the one in the row below it
    also share columns, but together are about twice as tall. Only sizeable
    pieces separated by a thin gap are joined, so specks next to an icon or
    strips of bezel along the border are not stitched into a digit.
    """
    def is_piece(b):
        return b[3] >= 0.2 * max_h and not touches_border(b, shape)

    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                xa, ya, wa, ha = boxes[i]
                xb, yb, wb, hb = boxes[j]
                if not (is_piece(boxes[i]) and is_piece(boxes[j])):
                    continue
                overlap = min(xa + wa, xb + wb) - max(xa, xb)
                gap = max(ya, yb) - min(ya + ha, yb + hb)
                x0, y0 = min(xa, xb), min(ya, yb)
                x1, y1 = max(xa + wa, xb + wb), max(ya + ha, yb + hb)
                if (overlap >= 0.5 * min(wa, wb) and gap <= 0.15 * max_h
                        and y1 - y0 <= 1.1 * max_h):
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return [tuple(b) for b in boxes]


def group_rows(boxes):
    """Group boxes into rows by vertical centre; rows top to bottom, boxes left to right."""
    boxes = sorted(boxes, key=lambda c: c[1] + c[3] / 2)
    rows = []
    for c in boxes:
        cy = c[1] + c[3] / 2
        if rows and abs(cy - rows[-1][0]) < 0.5 * c[3]:
            rows[-1][1].append(c)
        else:
            rows.append([cy, [c]])
    return [sorted(row, key=lambda c: c[0]) for _, row in rows]


def touches_border(box, shape, margin=2):
    x, y, bw, bh = box
    h, w = shape[:2]
    return x <= margin or y <= margin or x + bw >= w - margin or y + bh >= h - margin


def clean_row(row, shape):
    """
    Drop boxes that do not fit the row.

    The digits of one row are all the same height. A sliver of bezel or shadow
    along the image border can look digit-shaped on its own, but it is not the
    height of the digits next to it. Border boxes are only kept when they match
    the digits away from the border, so a digit clipped by Task 1 survives but a
    lone border sliver does not.
    """
    inner = [b for b in row if not touches_border(b, shape)]
    if not inner:
        return []
    ref_h = float(np.median([b[3] for b in inner]))
    return [b for b in row if abs(b[3] - ref_h) <= 0.25 * ref_h]


def find_lcd_digits(img):
    """Return digit boxes (x, y, w, h) as a list of rows, in reading order."""
    mask = lcd_segment_mask(img)
    h, w = mask.shape

    # Segments of one digit are separated by thin gaps; digits by much wider
    # ones. A small dilation joins each digit's segments into one blob.
    k = max(3, int(round(w * 0.025)))
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    joined = cv2.dilate(mask, element)
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)

    blobs = []
    for i in range(1, num_labels):
        # The LCD holds three rows of digits, so a real digit is well under
        # half the image height. Bezel edges and shadows along the border are
        # far taller (or wider, caught by the aspect test below).
        if stats[i, cv2.CC_STAT_HEIGHT] > 0.45 * h:
            continue
        # Tight box from the undilated segment pixels of this blob, using its
        # x and y distributions (column and row pixel counts, as in prac04).
        x, y, width, height, _ = stats[i]
        blob = (labels_im[y:y + height, x:x + width] == i) & (mask[y:y + height, x:x + width] > 0)
        x_distribution = np.sum(blob, axis=0)
        y_distribution = np.sum(blob, axis=1)
        xs, ys = np.nonzero(x_distribution)[0], np.nonzero(y_distribution)[0]
        if len(xs) == 0:
            continue
        blobs.append((x + xs[0], y + ys[0], xs[-1] + 1 - xs[0], ys[-1] + 1 - ys[0]))

    # Seven-segment digits are clearly taller than wide; icons like "OK" or
    # the heart are roughly square. Even a "1" is a solid bar, though, unlike
    # the hairline slivers of bezel left along the border.
    def digit_shaped(b):
        return 0.1 * b[3] <= b[2] <= b[3] / 1.4

    whole = [b for b in blobs if digit_shaped(b)]
    if not whole:
        return []
    blobs = merge_fragments(blobs, max(b[3] for b in whole), mask.shape)
    candidates = [b for b in blobs if digit_shaped(b)]

    # Drop small specks: the pulse digits are the smallest real ones (~60% of
    # the systolic height).
    max_h = max(c[3] for c in candidates)
    digits = [c for c in candidates if c[3] >= 0.4 * max_h]

    rows = [clean_row(row, mask.shape) for row in group_rows(digits)]
    return [row for row in rows if row]


def crop_box(img, box, pad_frac=0.01):
    """Crop (x, y, w, h) with a small padding, clipped to the image."""
    x, y, bw, bh = box
    pad = int(round(pad_frac * max(bw, bh)))
    h, w = img.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    return img[y0:y1, x0:x1]


def draw_lcd_debug(img, rows):
    vis = img.copy()
    thick = max(2, img.shape[1] // 300)
    n = 0
    for row in rows:
        for x, y, bw, bh in row:
            n += 1
            cv2.rectangle(vis, (int(x), int(y)), (int(x + bw), int(y + bh)), (0, 0, 255), thick)
            cv2.putText(vis, f"d{n}", (int(x), int(y) + 3 * thick + 10), cv2.FONT_HERSHEY_SIMPLEX,
                        img.shape[1] / 600, (0, 160, 0), thick)
    return vis


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


def liquid_response(img):
    """Per-pixel "thin coloured vertical line" strength, high on the liquid."""
    h, w = img.shape[:2]
    # Chroma (max - min channel) rather than HSV saturation: saturation
    # explodes on noisy dark pixels, which would make the black scale print
    # look "coloured".
    sat = (img.max(axis=2).astype(np.int16) - img.min(axis=2)).astype(np.uint8)
    sat = cv2.GaussianBlur(sat, (0, 0), 1.0)

    # The liquid is a thin, coloured vertical line; the scale print is black
    # (no chroma). After reversing the chroma image (as in prac02) the liquid
    # is a thin dark line, and a horizontal black-hat (closing minus image)
    # keeps only such narrow structures, i.e. the liquid.
    k = max(5, int(round(w * 0.05)))
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    response = cv2.morphologyEx(255 - sat, cv2.MORPH_BLACKHAT, element).astype(np.float32)
    # Average along the (near-vertical) line direction to suppress pixel noise
    # without widening the line.
    return cv2.GaussianBlur(response, (1, 0), sigmaX=0.1, sigmaY=max(1.0, h * 0.004))


def find_liquid_top(response, line, y_range=None):
    """
    Row of the top of the liquid column along the fitted line.

    y_range limits the search to the printed scale. The coloured bulb at the
    bottom of the tube responds far more strongly than the thin column, and if
    it is left in, the threshold below splits "bulb vs column" instead of
    "column vs empty tube".
    """
    h, w = response.shape
    a, b = line
    lo, hi = (0, h) if y_range is None else (max(0, int(y_range[0])), min(h, int(y_range[1])))
    if hi - lo < 10:
        return None

    # Sample the response along the fitted line, a few pixels either side.
    half = max(2, int(round(w * 0.006)))
    rows = np.arange(lo, hi)
    centres = np.clip(np.round(a * rows + b).astype(int), half, w - half - 1)
    profile = np.array([response[y, c - half:c + half + 1].mean() for y, c in zip(rows, centres)])
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (0, 0), max(1, h * 0.003)).ravel()

    # Otsu on the profile values splits "empty tube" from "liquid", so a
    # constant offset from noise or lighting does not matter.
    scaled = cv2.normalize(profile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Clip rare very high values (glare, any bit of bulb left in the range) so
    # they do not drag the threshold up.
    scaled = np.minimum(scaled, np.uint8(np.percentile(scaled, 95)))
    thr, _ = cv2.threshold(scaled.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    on = scaled > thr

    # Bridge short dropouts (glare, tick shadows) before looking for the run.
    gap = max(3, int(round(h * 0.01)))
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap))
    on = cv2.morphologyEx(on.astype(np.uint8).reshape(-1, 1), cv2.MORPH_CLOSE, element).ravel() > 0

    # The liquid is the longest "on" run; its top is the reading.
    best_len, best_top, run_start = 0, None, None
    for y, v in enumerate(np.append(on, False)):
        if v and run_start is None:
            run_start = y
        elif not v and run_start is not None:
            if y - run_start > best_len:
                best_len, best_top = y - run_start, run_start
            run_start = None
    return None if best_top is None else lo + best_top


def find_scale_labels(mask, liquid_x):
    """Vertical centres of the numbered labels on the left (degC) scale, top to bottom."""
    h, w = mask.shape
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Label digits are compact, glyph-sized blobs left of the tube. Tick marks
    # are thin and wide, so the aspect test removes them.
    glyphs = []
    for i in range(1, num_labels):
        x, y, bw, bh, _ = stats[i]
        if not (0.04 * w <= bh <= 0.15 * w):
            continue
        if bw > 1.2 * bh or x + bw > liquid_x - 0.08 * w:
            continue
        glyphs.append((centroids[i][1], bh))
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


def find_scale_ticks(mask, liquid_x, y_range):
    """
    Vertical positions of the tick marks on the left (degC) scale.

    The ticks form a band of short horizontal bars just left of the tube. The
    band is the run of mostly-dark columns nearest the tube (the tube itself is
    print-free); within it, each tick is a run of dark rows.
    """
    h, w = mask.shape
    y0, y1 = max(0, int(y_range[0])), min(h, int(y_range[1]))
    x_lo = max(0, int(liquid_x - 0.2 * w))
    x_hi = int(liquid_x)
    if y1 - y0 < 10 or x_hi - x_lo < 5:
        return [], (x_lo, x_lo)

    # x distribution (dark pixels per column, as in prac04) as a fraction.
    region = mask[y0:y1, x_lo:x_hi] > 0
    cols = np.sum(region, axis=0) / region.shape[0]
    thr = 0.5 * cols.max()
    # Walk left from the tube: skip the empty tube, then take the dark band.
    x = len(cols) - 1
    while x >= 0 and cols[x] < thr:
        x -= 1
    band_hi = x + 1
    while x >= 0 and cols[x] >= thr:
        x -= 1
    band_lo = x + 1
    if band_hi - band_lo < 2:
        return [], (x_lo, x_lo)

    # y distribution of the band: a tick row is mostly dark.
    band = mask[y0:y1, x_lo + band_lo:x_lo + band_hi] > 0
    rows = np.sum(band, axis=1) / band.shape[1]
    on = np.append(rows > 0.5, False)
    ticks, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        elif not v and start is not None:
            ticks.append(y0 + (start + y - 1) / 2)
            start = None
    return ticks, (x_lo + band_lo, x_lo + band_hi)


def analyse_thermometer(img):
    """
    Locate the liquid top and the scale around it, and choose the crop rows.

    Returns a dict with the liquid line, liquid top row, label and tick rows,
    the tick band columns and the crop rows (y0, y1).
    """
    h, w = img.shape[:2]
    response = liquid_response(img)
    line = fit_liquid_line(response)
    liquid_x = line[0] * h / 2 + line[1]

    mask = dark_print_mask(img)
    labels = find_scale_labels(mask, liquid_x)
    spacing = float(np.median(np.diff(labels))) if len(labels) >= 2 else None

    # The reading lies on the printed scale, so look for the liquid top only
    # between the end labels (half a label step of slack either side). This
    # keeps the coloured bulb below the scale out of the search.
    scale_range = (labels[0] - 0.5 * spacing, labels[-1] + 0.5 * spacing) if spacing else None
    top = find_liquid_top(response, line, scale_range)
    if top is None and scale_range is not None:
        top = find_liquid_top(response, line)
    if top is None:
        raise RuntimeError("liquid column not found")

    if spacing:
        # The pair of labels bracketing the liquid top (clamped to the ends).
        idx = int(np.searchsorted(labels, top))
        idx = min(max(idx, 1), len(labels) - 1)
        above, below = labels[idx - 1], labels[idx]
        # Centre the crop on the liquid top, and make it tall enough to show
        # both bracketing labels in full (a label is ~0.3 steps tall).
        margin = 0.3 * spacing
        half = max(top - (above - margin), (below + margin) - top)
    else:
        # No readable scale: fall back to a window centred on the liquid top.
        half = 0.3 * w
    y0 = int(max(0, round(top - half)))
    y1 = int(min(h, round(top + half)))

    ticks, band = find_scale_ticks(mask, liquid_x, (y0, y1))
    return {"line": line, "top": int(top), "labels": labels, "spacing": spacing,
            "ticks": ticks, "tick_band": band, "crop": (y0, y1)}


def draw_thermo_debug(img, info):
    vis = img.copy()
    h, w = img.shape[:2]
    thick = max(2, w // 300)
    a, b = info["line"]
    cv2.line(vis, (int(round(b)), 0), (int(round(a * h + b)), h), (255, 0, 0), 1)
    for y in info["labels"]:
        cv2.line(vis, (0, int(y)), (int(w * 0.15), int(y)), (0, 160, 0), thick)
    bx0, bx1 = info["tick_band"]
    for y in info["ticks"]:
        cv2.line(vis, (bx0, int(y)), (bx1, int(y)), (255, 0, 255), 1)
    top = info["top"]
    cv2.line(vis, (0, top), (w, top), (0, 0, 255), thick)
    y0, y1 = info["crop"]
    cv2.rectangle(vis, (0, y0), (w - 1, y1 - 1), (0, 200, 255), thick)
    return vis


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def is_thermometer(path, img):
    """Device type from the file name (lcdX / thermoX), else from the shape."""
    name = os.path.basename(path).lower()
    if name.startswith("thermo"):
        return True
    if name.startswith("lcd"):
        return False
    h, w = img.shape[:2]
    return h / w >= THERMO_MIN_ASPECT


def list_images(image_path):
    if os.path.isfile(image_path):
        return [image_path]
    return sorted(p for p in glob.glob(os.path.join(image_path, "*"))
                  if p.lower().endswith(IMAGE_EXTS))


def read_settings(config):
    """Task 2 settings from config.txt, falling back to the defaults."""
    config = config or {}
    return {"debug": int(config.get("task2_debug", DEFAULT_DEBUG)) != 0}


def process_lcd(img, out_dir):
    rows = find_lcd_digits(img)
    sizes = ", ".join(str(len(r)) for r in rows)
    print(f"  bpm: {sum(len(r) for r in rows)} digit(s) in {len(rows)} row(s) [{sizes}]")
    if len(rows) != 3:
        print("  warning: expected 3 rows (SYS, DIA, PULSE)")
    boxes = [box for row in rows for box in row]
    for i, box in enumerate(boxes, start=1):
        save_output(os.path.join(out_dir, f"d{i}.png"), crop_box(img, box), output_type='image')
    return draw_lcd_debug(img, rows)


def process_thermo(img, out_dir):
    h = img.shape[0]
    info = analyse_thermometer(img)
    y0, y1 = info["crop"]
    print(f"  thermo: liquid top at row {info['top']} ({100 * info['top'] / h:.1f}% down)")
    if info["spacing"]:
        print(f"  {len(info['labels'])} scale labels, {info['spacing']:.1f} px apart (10 degC)")
    else:
        print("  warning: scale labels not found, cropping around the liquid top only")
    if len(info["ticks"]) >= 2:
        tick_step = float(np.median(np.diff(info["ticks"])))
        print(f"  {len(info['ticks'])} tick marks in the crop, {tick_step:.1f} px apart (1 degC)")
    print(f"  crop rows {y0}-{y1}")
    save_output(os.path.join(out_dir, "t.png"), img[y0:y1, :], output_type='image')
    return draw_thermo_debug(img, info)


def run_task2(image_path, config):
    settings = read_settings(config)
    images = list_images(image_path)
    print(f"Task 2: {len(images)} image(s) in {image_path} ({settings})")

    for path in images:
        img = cv2.imread(path)
        if img is None:
            print(f"Could not read image: {path}")
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(OUTPUT_DIR, name)
        print(f"{os.path.basename(path)}:")

        try:
            if is_thermometer(path, img):
                vis = process_thermo(img, out_dir)
            else:
                vis = process_lcd(img, out_dir)
            if settings["debug"]:
                save_output(os.path.join(OUTPUT_DIR, "debug", f"debug_{name}.png"), vis,
                            output_type='image')
        except Exception as e:
            # One bad image must not stop the rest of the batch.
            print(f"Task 2 failed on {path}: {e}")
