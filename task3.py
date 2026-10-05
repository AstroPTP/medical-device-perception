

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
Task 3: turn the segmented reading (output of Task 2) into numbers.

  - BPM lcdX/dZ.png  -> dZ.txt, the digit (0-9) recognised by a small CNN.
  - Thermo t.png     -> t.txt, the temperature in whole degC.

Digit recognition uses DigitCNN, trained by train_task3.py (weights in
data/task3_digits.pt). The checkpoint holds two copies of the same network:
  - "lcd"   : seven-segment LCD digits (BPM readings)
  - "print" : printed scale numbers (thermometer labels "20", "10", ...)

Thermometer calculation, all on t.png:
  1. Find the liquid column and its top (reusing the Task 2 detectors).
  2. Find the 1 degC tick marks next to the tube, and their spacing.
  3. Find the printed degC labels left of the ticks and read them with the
     "print" network. Only the order of labels is assumed (they go down in
     steps of 10 degC), so all labels are decoded jointly: every possible
     value for the top label is scored against what the network read, and the
     best one wins. One misread glyph is outvoted by the others.
  4. Each tick gets its value from the labels; the reading is a linear
     interpolation between the two ticks either side of the liquid top.

Set "task3_debug: 1" in config.txt to also save output/task3/debug/<name>.png,
showing the liquid top, labels and ticks used for the thermometer reading.
"""

import glob
import math
import os
import re

import cv2
import numpy as np
import torch
from torch import nn

import task2

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "data", "task3_digits.pt")
OUTPUT_DIR = os.path.join(HERE, "output", "task3")

DEFAULT_DEBUG = 0      # 1 = also save a debug image per thermometer

# Network input: digits are fitted into this box keeping their aspect ratio,
# so a narrow "1" stays narrow (its shape is a big part of what makes it a 1).
DIGIT_H, DIGIT_W = 48, 32

# The degC labels printed on the thermometer scale, top to bottom.
SCALE_LABELS = list(range(50, -30, -10))

_models = {}


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
# Digit classifier
# --------------------------------------------------------------------------- #

class DigitCNN(nn.Module):
    """
    Small VGG-style CNN for 48x32 single-digit images (10 classes).

    Three conv blocks (two 3x3 convs + max pool, then one) take the input
    down to a 6x4 map of 128 features, followed by a dropout-regularised
    fully connected classifier. ~0.5M parameters: plenty for 10 clean,
    centred glyph classes, and fast on a CPU.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        def block(c_in, c_out):
            return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                    nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(
            *block(1, 32), *block(32, 32), nn.MaxPool2d(2),     # 24 x 16
            *block(32, 64), *block(64, 64), nn.MaxPool2d(2),    # 12 x 8
            *block(64, 128), nn.MaxPool2d(2),                   # 6 x 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(128 * (DIGIT_H // 8) * (DIGIT_W // 8), 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def prepare_digit(img):
    """
    Normalise one digit crop into the network input (DIGIT_H x DIGIT_W, float32).

    Ink (dark LCD segments or dark print) becomes bright on a black canvas,
    with the contrast stretched between the 1st and 99th percentile so the
    lighting and LCD tint do not matter. The crop is scaled to fit the canvas
    without changing its aspect ratio and centred. train_task3.py uses this
    same function, so training and inference see identical inputs.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    lo, hi = np.percentile(gray, (1, 99))
    # A floor on the range stops a crop that is nearly all ink (a "1") from
    # having its noise stretched to full contrast.
    ink = np.clip((hi - gray) / max(hi - lo, 40.0), 0.0, 1.0)

    h, w = ink.shape
    scale = min((DIGIT_H - 4) / h, (DIGIT_W - 4) / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = cv2.resize(ink, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((DIGIT_H, DIGIT_W), np.float32)
    y0, x0 = (DIGIT_H - nh) // 2, (DIGIT_W - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    return canvas


def load_models():
    """Load both networks once and reuse them (Task 4 calls this per image)."""
    if not _models:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model weights not found: {MODEL_PATH} (run train_task3.py)")
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        for name in ("lcd", "print"):
            model = DigitCNN()
            model.load_state_dict(checkpoint[name])
            model.eval()
            _models[name] = model
        print(f"Loaded digit classifiers: {MODEL_PATH}")
    return _models


def classify_digits(crops, kind="lcd"):
    """
    Class probabilities (N x 10) for a list of digit crops.

    kind is "lcd" for seven-segment digits or "print" for printed numbers.
    """
    if len(crops) == 0:
        return np.zeros((0, 10), np.float32)
    model = load_models()[kind]
    batch = np.stack([prepare_digit(c) for c in crops])[:, None]
    with torch.no_grad():
        logits = model(torch.from_numpy(batch))
    return torch.softmax(logits, dim=1).numpy()


def recognise_digits(crops):
    """Recognised digit (int) and its confidence for each seven-segment crop."""
    probs = classify_digits(crops, "lcd")
    return [(int(p.argmax()), float(p.max())) for p in probs]


# --------------------------------------------------------------------------- #
# Thermometer
# --------------------------------------------------------------------------- #

def liquid_profile(response, line, w):
    """Liquid response along the fitted tube line, one value per row (as in Task 2)."""
    h = response.shape[0]
    a, b = line
    half = max(2, int(round(w * 0.006)))
    rows = np.arange(h)
    centres = np.clip(np.round(a * rows + b).astype(int), half, w - half - 1)
    profile = np.array([response[y, c - half:c + half + 1].mean() for y, c in zip(rows, centres)])
    return cv2.GaussianBlur(profile.reshape(-1, 1), (0, 0), 1.0).ravel()


def find_liquid_top(profile, deg_px):
    """
    Row of the top of the liquid column: the best "empty above, liquid below" split.

    The column can be faint and get stronger further down, which makes a
    global threshold (Task 2's method) put the top too low. Instead the
    profile is fitted with a step function, one level above row y and another
    below it, and y is chosen to minimise the squared error (Otsu's idea, but
    along the tube rather than over the histogram). The column tip is faint
    and tapered, so the split is then moved up while the profile stays above
    30% of the way from the empty level to the liquid level (by at most three
    ticks, so noise above the column cannot drag it far).
    """
    n = len(profile)
    if n < 10:
        return None
    c1 = np.cumsum(profile)
    c2 = np.cumsum(profile ** 2)
    best_y, best_sse = None, None
    for y in range(max(1, int(0.05 * n)), int(0.95 * n)):
        sum_a, sum_b = c1[y - 1], c1[-1] - c1[y - 1]
        sq_a, sq_b = c2[y - 1], c2[-1] - c2[y - 1]
        if sum_b / (n - y) <= sum_a / y:
            continue                        # liquid must be the stronger side
        sse = sq_a - sum_a ** 2 / y + sq_b - sum_b ** 2 / (n - y)
        if best_sse is None or sse < best_sse:
            best_y, best_sse = y, sse
    if best_y is None:
        return None

    lo = float(np.median(profile[:best_y]))
    hi = float(np.median(profile[best_y:]))
    level = lo + 0.3 * (hi - lo)
    top = best_y
    while top > 0 and best_y - top < 3 * deg_px and profile[top - 1] >= level:
        top -= 1
    return top


def find_ticks(mask, liquid_x, h):
    """
    1 degC tick marks left of the tube: their rows, the tick band and spacing.

    Uses the Task 2 tick finder over the whole crop. Spurious ticks (a merged
    pair, a gap) are tolerated because the spacing is the median gap.
    """
    ticks, band = task2.find_scale_ticks(mask, liquid_x, (0, h))
    step = float(np.median(np.diff(ticks))) if len(ticks) >= 3 else None
    return ticks, band, step


def find_label_rows(img, mask, band_x0, tick_step):
    """
    Printed degC labels left of the tick band.

    Returns a list of rows, top to bottom. Each row is a dict with its centre
    row "y", its digit glyph boxes "glyphs" (left to right) and "minus"
    (whether a minus sign was found in front of the digits).
    Glyph size is judged against the tick spacing when it is known (a label is
    ~3 ticks tall), so this does not depend on the crop's resolution.
    """
    h, w = mask.shape
    if tick_step:
        min_h, max_h = 1.5 * tick_step, 5.0 * tick_step
    else:
        min_h, max_h = 0.04 * w, 0.2 * w

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    glyphs, dashes = [], []
    for i in range(1, num_labels):
        x, y, bw, bh, _ = stats[i]
        if x + bw > band_x0:
            continue
        if min_h <= bh <= max_h and bw <= 1.2 * bh:
            glyphs.append((x, y, bw, bh))
        elif bw >= 1.5 * bh:
            dashes.append((x, y, bw, bh))
    if not glyphs:
        return []

    # Glyphs of one label ("20") share a row. Drop odd-sized ones first (a
    # "C" of the degC heading is a different size from the numbers).
    ref_h = float(np.median([g[3] for g in glyphs]))
    glyphs = [g for g in glyphs if abs(g[3] - ref_h) <= 0.3 * ref_h]
    rows = []
    for g in sorted(glyphs, key=lambda g: g[1] + g[3] / 2):
        cy = g[1] + g[3] / 2
        if rows and abs(cy - rows[-1]["y"]) < 0.5 * ref_h:
            rows[-1]["glyphs"].append(g)
            rows[-1]["y"] = float(np.mean([b[1] + b[3] / 2 for b in rows[-1]["glyphs"]]))
        else:
            rows.append({"y": cy, "glyphs": [g]})

    for row in rows:
        # Labels have at most two digits, written up against the tick marks:
        # keep the two glyphs nearest the ticks.
        row["glyphs"] = sorted(row["glyphs"], key=lambda g: g[0])[-2:]
        left = row["glyphs"][0][0]
        # A minus sign is a short, flat dash just in front of the digits.
        row["minus"] = any(
            abs((d[1] + d[3] / 2) - row["y"]) < 0.3 * ref_h and d[3] < 0.4 * ref_h
            and 0 <= left - (d[0] + d[2]) < 1.0 * ref_h and d[2] < 1.2 * ref_h
            for d in dashes)
    return rows


def decode_labels(img, rows, deg_px):
    """
    Read the label values jointly.

    Every label is 10 degC below the one above it, so the whole column of
    labels is fixed by the value of the top one. Each candidate top value is
    scored by how well the network's reading of every glyph (and the minus
    signs) agrees with the strings it predicts; the best candidate wins.
    Returns ([value per row], confidence of the winning candidate).
    """
    # Which label slot each row is in, counted from the top row; this allows
    # for a label that was missed between two found ones.
    ys = np.array([r["y"] for r in rows])
    steps = np.round((ys - ys[0]) / (10 * deg_px)).astype(int)

    crops, owners = [], []
    for k, row in enumerate(rows):
        for g in row["glyphs"]:
            crops.append(task2.crop_box(img, g, pad_frac=0.08))
            owners.append(k)
    probs = classify_digits(crops, "print")
    per_row = [[p for p, o in zip(probs, owners) if o == k] for k in range(len(rows))]

    eps = 1e-4
    scores = {}
    for top in SCALE_LABELS:
        values = [top - 10 * s for s in steps]
        if min(values) < SCALE_LABELS[-1]:
            continue
        score = 0.0
        for row, glyph_probs, value in zip(rows, per_row, values):
            text = str(abs(value))
            # Align from the right: the units digit is next to the tick.
            for j in range(1, max(len(text), len(glyph_probs)) + 1):
                if j <= len(text) and j <= len(glyph_probs):
                    score += math.log(glyph_probs[-j][int(text[-j])] + eps)
                else:
                    score += math.log(0.02)     # a glyph missing or extra
            if row["minus"] != (value < 0):
                score += math.log(0.25)         # the dash is easy to miss
        scores[top] = score
    if not scores:
        return None, 0.0
    best = max(scores, key=scores.get)
    total = sum(math.exp(s - scores[best]) for s in scores.values())
    return [best - 10 * s for s in steps], 1.0 / total


def interpolate_reading(top_y, label_ys, label_values, ticks, deg_px):
    """
    Temperature at row top_y.

    The labels give a linear row -> degC map. Each tick near the liquid top is
    snapped to its whole degree with that map, and the reading is interpolated
    linearly between the tick just below and the tick just above the top. If
    the ticks are unusable, the label map is used directly.
    """
    # Least squares offset for the line T = c - y / deg_px through the labels.
    c = float(np.mean(np.asarray(label_values) + np.asarray(label_ys) / deg_px))

    def degrees(y):
        return c - y / deg_px

    # A label's centre is only roughly level with its tick, but the ticks sit
    # exactly on whole degrees: shift the map by the ticks' median offset from
    # the nearest whole degree.
    if len(ticks):
        offsets = [degrees(y) - round(degrees(y)) for y in ticks]
        c -= float(np.median(offsets))

    tick_values = []
    for y in ticks:
        t = degrees(y)
        if abs(t - round(t)) < 0.35:      # ignore anything not on a tick position
            tick_values.append((y, round(t)))
    below = [tv for tv in tick_values if tv[0] >= top_y]
    above = [tv for tv in tick_values if tv[0] < top_y]
    if below and above:
        y_lo, t_lo = min(below, key=lambda tv: tv[0])
        y_hi, t_hi = max(above, key=lambda tv: tv[0])
        if t_hi > t_lo and y_lo > y_hi and t_hi - t_lo <= 2:
            return t_lo + (y_lo - top_y) / (y_lo - y_hi) * (t_hi - t_lo), "ticks"
    return degrees(top_y), "labels"


def read_thermometer(img):
    """
    Temperature (float degC) shown on a segmented scale image (t.png).

    Returns a dict with the reading and everything found along the way, for
    printing and the debug image. Raises RuntimeError if the liquid or the
    labels cannot be found.
    """
    h, w = img.shape[:2]
    response = task2.liquid_response(img)
    line = task2.fit_liquid_line(response)
    liquid_x = line[0] * h / 2 + line[1]

    mask = task2.dark_print_mask(img)
    ticks, band, tick_step = find_ticks(mask, liquid_x, h)
    band_x0 = band[0] if band[1] > band[0] else liquid_x - 0.1 * w
    rows = find_label_rows(img, mask, band_x0, tick_step)
    if not rows:
        raise RuntimeError("no scale labels found")

    # Pixels per degC: from the ticks, else from the label spacing (10 degC).
    if tick_step:
        deg_px = tick_step
    elif len(rows) >= 2:
        deg_px = float(np.median(np.diff([r["y"] for r in rows]))) / 10
    else:
        raise RuntimeError("scale spacing not found (need ticks or two labels)")

    top = find_liquid_top(liquid_profile(response, line, w), deg_px)
    if top is None:
        top = task2.find_liquid_top(response, line)     # Task 2's threshold method
    if top is None:
        raise RuntimeError("liquid column not found")

    values, confidence = decode_labels(img, rows, deg_px)
    if values is None:
        raise RuntimeError("labels could not be read")
    temp, method = interpolate_reading(top, [r["y"] for r in rows], values, ticks, deg_px)
    return {"temp": temp, "method": method, "top": top, "line": line, "ticks": ticks,
            "band": band, "deg_px": deg_px, "rows": rows, "values": values,
            "label_conf": confidence}


def draw_thermo_debug(img, info):
    vis = img.copy()
    h, w = img.shape[:2]
    thick = max(1, w // 300)
    a, b = info["line"]
    cv2.line(vis, (int(round(b)), 0), (int(round(a * h + b)), h), (255, 0, 0), 1)
    bx0, bx1 = info["band"]
    for y in info["ticks"]:
        cv2.line(vis, (bx0, int(y)), (bx1, int(y)), (255, 0, 255), 1)
    for row, value in zip(info["rows"], info["values"]):
        for x, y, bw, bh in row["glyphs"]:
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 160, 0), thick)
        x = row["glyphs"][-1][0] + row["glyphs"][-1][2]
        cv2.putText(vis, str(value), (x + 2, int(row["y"])), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.4, w / 800), (0, 160, 0), thick)
    top = info["top"]
    cv2.line(vis, (0, top), (w, top), (0, 0, 255), thick)
    cv2.putText(vis, f"{info['temp']:.1f} C", (w // 2 + 10, max(15, top - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.4, w / 800), (0, 0, 255), thick)
    return vis


def round_half_up(x):
    """Round to the nearest integer, halves up (Python's round() rounds to even)."""
    return int(math.floor(x + 0.5))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def digit_number(path):
    """'d12.png' -> 12, so d10 sorts after d9."""
    match = re.search(r"(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else 0


def read_settings(config):
    """Task 3 settings from config.txt, falling back to the defaults."""
    config = config or {}
    return {"debug": int(config.get("task3_debug", DEFAULT_DEBUG)) != 0}


def process_lcd(in_dir, out_dir):
    # Only d1.png, d2.png, ... (not e.g. a stray debug_*.png).
    paths = sorted((p for p in glob.glob(os.path.join(in_dir, "*.png"))
                    if re.fullmatch(r"d\d+\.png", os.path.basename(p))), key=digit_number)
    if not paths:
        print("  no digit images (d1.png, d2.png, ...) found")
        return
    crops, names = [], []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  could not read {path}")
            continue
        crops.append(img)
        names.append(os.path.splitext(os.path.basename(path))[0])
    results = recognise_digits(crops)
    for name, (digit, conf) in zip(names, results):
        print(f"  {name}: {digit} ({conf:.2f})")
        save_output(os.path.join(out_dir, f"{name}.txt"), str(digit), output_type='txt')
    print(f"  bpm digits: {''.join(str(d) for d, _ in results)}")


def process_thermo(in_dir, out_dir, debug_path=None):
    img = cv2.imread(os.path.join(in_dir, "t.png"))
    if img is None:
        print(f"  could not read {os.path.join(in_dir, 't.png')}")
        return
    info = read_thermometer(img)
    labels = ", ".join(f"{v} @ row {r['y']:.0f}" for v, r in zip(info["values"], info["rows"]))
    print(f"  liquid top at row {info['top']}, {len(info['ticks'])} ticks, "
          f"{info['deg_px']:.1f} px per degC")
    print(f"  labels: {labels} (confidence {info['label_conf']:.2f})")
    reading = round_half_up(info["temp"])
    print(f"  thermo: {info['temp']:.2f} degC (from {info['method']}) -> {reading}")
    save_output(os.path.join(out_dir, "t.txt"), str(reading), output_type='txt')
    if debug_path:
        save_output(debug_path, draw_thermo_debug(img, info), output_type='image')


def run_task3(image_path, config):
    settings = read_settings(config)
    # Reading folders are lcdX/ and thermoX/; anything else (e.g. a debug/
    # folder left by Task 2) is skipped.
    subdirs = sorted(d for d in glob.glob(os.path.join(image_path, "*"))
                     if os.path.isdir(d) and re.match(r"(lcd|thermo)", os.path.basename(d).lower()))
    print(f"Task 3: {len(subdirs)} reading(s) in {image_path} ({settings})")

    for in_dir in subdirs:
        name = os.path.basename(os.path.normpath(in_dir))
        out_dir = os.path.join(OUTPUT_DIR, name)
        print(f"{name}:")
        try:
            if name.lower().startswith("thermo"):
                debug_path = (os.path.join(OUTPUT_DIR, "debug", f"debug_{name}.png")
                              if settings["debug"] else None)
                process_thermo(in_dir, out_dir, debug_path)
            else:
                process_lcd(in_dir, out_dir)
        except Exception as e:
            # One bad reading must not stop the rest of the batch.
            print(f"Task 3 failed on {in_dir}: {e}")
