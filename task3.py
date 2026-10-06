

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
# Last Modified: 2026-10-07

"""
Task 3: turn the segmented reading (output of Task 2) into numbers.

  - BPM lcdX/dZ.png  -> dZ.txt, the digit (0-9) recognised by a small CNN
  - Thermo t.png     -> t.txt, the temperature in whole degC

Digit recognition uses DigitCNN, trained by train_task3.py (weights in
data/task3_digits.pt). The checkpoint holds two copies of the same network:
  - "lcd"   : seven-segment LCD digits (BPM readings)
  - "print" : printed scale numbers (thermometer labels "20", "10", ...)

Thermometer calculation, all on t.png:
  1. Find the liquid column and its top (reusing the Task 2 detectors)
  2. Find the 1 degC tick marks next to the tube, and their spacing
  3. Find the printed degC labels left of the ticks and read them with the
     "print" network. Only the order of labels is assumed (they go down in
     steps of 10 degC), so all labels are decoded jointly: every possible
     value for the top label is scored against what the network read, and the
     best one wins. One misread glyph is outvoted by the others
  4. Each tick gets its value from the labels; the reading is a linear
     interpolation between the two ticks either side of the liquid top

Set "task3_debug: 1" in config.txt to also save output/task3/debug/<name>.png,
showing the liquid top, labels and ticks used for the thermometer reading
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

HERE = os.path.dirname(os.path.abspath(__file__)) # Absolute path of the directory containing this script, so paths resolve from any working directory
MODEL_PATH = os.path.join(HERE, "data", "task3_digits.pt") # Digit classifier weights written by train_task3.py
OUTPUT_DIR = os.path.join(HERE, "output", "task3") # Output folder for the recognised readings

# 1 = also save a debug image per thermometer
DEFAULT_DEBUG = 0

# Network input: digits are fitted into this box keeping their aspect ratio,
# so a narrow "1" stays narrow (its shape is a big part of what makes it a 1)
DIGIT_H, DIGIT_W = 48, 32

# The degC labels printed on the thermometer scale, top to bottom
SCALE_LABELS = list(range(50, -30, -10))

# Cached networks by name ("lcd", "print"), loaded on first use by load_models
_models = {}

def save_output(output_path, content, output_type='txt'):
    """
    save_output: Save text or an image to output_path, creating its folder if needed

    :param output_path (str): destination file path
    :param content (str or np.ndarray): text to write, or an OpenCV image
    :param output_type (str): 'txt' for text or 'image' for an image
    :return: None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_type == 'txt':
        with open(output_path, 'w') as file:
            file.write(content)
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
    centred glyph classes, and fast on a CPU
    """

    def __init__(self, num_classes=10):
        """
        __init__: Build the convolutional feature extractor and the classifier

        :param num_classes (int): number of output classes
        :return: None
        """
        super().__init__()

        def block(in_channels, out_channels):
            """
            block: Build one 3x3 convolution + batch norm + ReLU layer group

            :param in_channels (int): input feature channels
            :param out_channels (int): output feature channels
            :return: list of the three layers
            """
            return [nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]

        # The layer order must match the saved checkpoint
        self.features = nn.Sequential(
            *block(1, 32),
            *block(32, 32),
            nn.MaxPool2d(2),        # 24 x 16
            *block(32, 64),
            *block(64, 64),
            nn.MaxPool2d(2),        # 12 x 8
            *block(64, 128),
            nn.MaxPool2d(2),        # 6 x 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * (DIGIT_H // 8) * (DIGIT_W // 8), 128),     # 128 features x 6 x 4 map
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, batch):
        """
        forward: Compute the class scores for a batch of digit images

        :param batch (torch.Tensor): inputs of shape (N, 1, DIGIT_H, DIGIT_W)
        :return: class logits of shape (N, num_classes) (torch.Tensor)
        """
        return self.classifier(self.features(batch))

def prepare_digit(img):
    """
    prepare_digit: Normalise one digit crop into the network input (DIGIT_H x DIGIT_W,
    float32)

    Ink (dark LCD segments or dark print) becomes bright on a black canvas,
    with the contrast stretched between the 1st and 99th percentile so the
    lighting and LCD tint do not matter. The crop is scaled to fit the canvas
    without changing its aspect ratio and centred. train_task3.py uses this
    same function, so training and inference see identical inputs

    :param img (np.ndarray): digit crop, BGR or grayscale
    :return: network input with values in [0, 1] (np.ndarray, float32)
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    low, high = np.percentile(gray, (1, 99))

    # A floor on the range stops a crop that is nearly all ink (a "1") from
    # having its noise stretched to full contrast.
    ink = np.clip((high - gray) / max(high - low, 40.0), 0.0, 1.0)

    # Scale to fit inside the canvas with a 2 px border, keeping the aspect ratio
    height, width = ink.shape
    scale = min((DIGIT_H - 4) / height, (DIGIT_W - 4) / width)
    new_height, new_width = max(1, int(round(height * scale))), max(1, int(round(width * scale)))
    small = cv2.resize(ink, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Centre the scaled digit on a black canvas
    canvas = np.zeros((DIGIT_H, DIGIT_W), np.float32)
    top, left = (DIGIT_H - new_height) // 2, (DIGIT_W - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = small

    return canvas

def load_models():
    """
    load_models: Load both networks once and reuse them (Task 4 calls this per image)

    :return: dict mapping "lcd" and "print" to their DigitCNN in evaluation mode
    """
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
    classify_digits: Compute class probabilities for a list of digit crops

    :param crops (list): digit crops (np.ndarray), BGR or grayscale
    :param kind (str): "lcd" for seven-segment digits or "print" for printed numbers
    :return: probabilities of shape (N, 10) (np.ndarray)
    """
    if len(crops) == 0:
        return np.zeros((0, 10), np.float32)
    
    model = load_models()[kind]
    # Stack the prepared crops and add the channel axis: (N, 1, H, W)
    batch = np.stack([prepare_digit(crop) for crop in crops])[:, None]

    with torch.no_grad():
        logits = model(torch.from_numpy(batch))

    return torch.softmax(logits, dim=1).numpy()

def recognise_digits(crops):
    """
    recognise_digits: Recognise each seven-segment digit crop

    :param crops (list): digit crops (np.ndarray)
    :return: list of (digit (int), confidence (float)) per crop
    """
    probabilities = classify_digits(crops, "lcd")
    return [(int(probs.argmax()), float(probs.max())) for probs in probabilities]

# --------------------------------------------------------------------------- #
# Thermometer
# --------------------------------------------------------------------------- #

def liquid_profile(response, line, width):
    """
    liquid_profile: Sample the liquid response along the fitted tube line, one value
    per row (as in Task 2)

    :param response (np.ndarray): liquid response image
    :param line (tuple): (slope, intercept) of the liquid line
    :param width (int): image width
    :return: smoothed response per row (np.ndarray)
    """
    height = response.shape[0]
    slope, intercept = line

    # Average a few pixels either side of the line in every row
    half_width = max(2, int(round(width * 0.006)))
    rows = np.arange(height)

    centre_cols = np.clip(np.round(slope * rows + intercept).astype(int), half_width, width - half_width - 1)
    profile = np.array([response[row, col - half_width:col + half_width + 1].mean()
                        for row, col in zip(rows, centre_cols)])
    
    return cv2.GaussianBlur(profile.reshape(-1, 1), (0, 0), 1.0).ravel()

def find_liquid_top(profile, px_per_degree):
    """
    find_liquid_top: Find the row of the top of the liquid column, the best "empty
    above, liquid below" split

    The column can be faint and get stronger further down, which makes a
    global threshold (Task 2's method) put the top too low. Instead the
    profile is fitted with a step function, one level above row y and another
    below it, and y is chosen to minimise the squared error (Otsu's idea, but
    along the tube rather than over the histogram). The column tip is faint
    and tapered, so the split is then moved up while the profile stays above
    30% of the way from the empty level to the liquid level (by at most three
    ticks, so noise above the column cannot drag it far)

    :param profile (np.ndarray): liquid response per row
    :param px_per_degree (float): pixels per degC (tick spacing)
    :return: row of the liquid top (int), or None if not found
    """
    length = len(profile)
    if length < 10:
        return None
    
    # Cumulative sums give the mean and squared error of any split in O(1)
    cum_sum = np.cumsum(profile)
    cum_sq_sum = np.cumsum(profile ** 2)
    best_y, best_error = None, None

    # Try every split except the outer 5% at each end
    for split_y in range(max(1, int(0.05 * length)), int(0.95 * length)):
        sum_above, sum_below = cum_sum[split_y - 1], cum_sum[-1] - cum_sum[split_y - 1]
        sq_above, sq_below = cum_sq_sum[split_y - 1], cum_sq_sum[-1] - cum_sq_sum[split_y - 1]

        # Liquid must be the stronger side
        if sum_below / (length - split_y) <= sum_above / split_y:
            continue

        # Squared error of a two-level step fit
        error = sq_above - sum_above ** 2 / split_y + sq_below - sum_below ** 2 / (length - split_y)
        if best_error is None or error < best_error:
            best_y, best_error = split_y, error
    if best_y is None:
        return None

    empty_level = float(np.median(profile[:best_y]))
    liquid_level = float(np.median(profile[best_y:]))
    level = empty_level + 0.3 * (liquid_level - empty_level)
    top = best_y

    # Move up over the faint tip, by at most three ticks
    while top > 0 and best_y - top < 3 * px_per_degree and profile[top - 1] >= level:
        top -= 1
    return top

def find_ticks(mask, liquid_x, height):
    """
    find_ticks: Find the 1 degC tick marks left of the tube, the tick band and the spacing

    Uses the Task 2 tick finder over the whole crop. Spurious ticks (a merged
    pair, a gap) are tolerated because the spacing is the median gap

    :param mask (np.ndarray): dark print mask
    :param liquid_x (float): column of the liquid line
    :param height (int): image height
    :return: (tick rows (list), (band_left, band_right) columns, tick spacing (float) or None)
    """
    ticks, band = task2.find_scale_ticks(mask, liquid_x, (0, height))
    tick_step = float(np.median(np.diff(ticks))) if len(ticks) >= 3 else None
    return ticks, band, tick_step

def find_label_rows(img, mask, band_left, tick_step):
    """
    find_label_rows: Find the printed degC labels left of the tick band

    Glyph size is judged against the tick spacing when it is known (a label is
    ~3 ticks tall), so this does not depend on the crop's resolution

    :param img (np.ndarray): BGR scale image
    :param mask (np.ndarray): dark print mask
    :param band_left (float): left column of the tick band; labels lie left of it
    :param tick_step (float): tick spacing in pixels, or None if unknown
    :return: list of rows, top to bottom. Each row is a dict with its centre row "y",
             its digit glyph boxes "glyphs" (left to right) and "minus" (whether a
             minus sign was found in front of the digits)
    """
    height, width = mask.shape
    # Allowed glyph height range
    if tick_step:
        min_height, max_height = 1.5 * tick_step, 5.0 * tick_step
    else:
        min_height, max_height = 0.04 * width, 0.2 * width

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    glyphs, dashes = [], []

    # Sort blobs left of the band into digit glyphs and flat dashes (minus signs)
    for label in range(1, num_labels):
        box_x, box_y, box_width, box_height, _ = stats[label]
        if box_x + box_width > band_left:
            continue
        if min_height <= box_height <= max_height and box_width <= 1.2 * box_height:
            glyphs.append((box_x, box_y, box_width, box_height))
        elif box_width >= 1.5 * box_height:
            dashes.append((box_x, box_y, box_width, box_height))
    if not glyphs:
        return []

    # Glyphs of one label ("20") share a row. Drop odd-sized ones first (a
    # "C" of the degC heading is a different size from the numbers)
    ref_height = float(np.median([glyph[3] for glyph in glyphs]))
    glyphs = [glyph for glyph in glyphs if abs(glyph[3] - ref_height) <= 0.3 * ref_height]
    rows = []

    for glyph in sorted(glyphs, key=lambda glyph: glyph[1] + glyph[3] / 2):
        centre_y = glyph[1] + glyph[3] / 2
        if rows and abs(centre_y - rows[-1]["y"]) < 0.5 * ref_height:
            rows[-1]["glyphs"].append(glyph)
            rows[-1]["y"] = float(np.mean([box[1] + box[3] / 2 for box in rows[-1]["glyphs"]]))
        else:
            rows.append({"y": centre_y, "glyphs": [glyph]})

    for row in rows:
        # Labels have at most two digits, written up against the tick marks:
        # keep the two glyphs nearest the ticks
        row["glyphs"] = sorted(row["glyphs"], key=lambda glyph: glyph[0])[-2:]
        digits_left = row["glyphs"][0][0]

        # A minus sign is a short, flat dash just in front of the digits
        row["minus"] = any(
            abs((dash[1] + dash[3] / 2) - row["y"]) < 0.3 * ref_height and dash[3] < 0.4 * ref_height
            and 0 <= digits_left - (dash[0] + dash[2]) < 1.0 * ref_height and dash[2] < 1.2 * ref_height
            for dash in dashes)
        
    return rows

def decode_labels(img, rows, px_per_degree):
    """
    decode_labels: Read the label values jointly

    Every label is 10 degC below the one above it, so the whole column of
    labels is fixed by the value of the top one. Each candidate top value is
    scored by how well the network's reading of every glyph (and the minus
    signs) agrees with the strings it predicts; the best candidate wins

    :param img (np.ndarray): BGR scale image
    :param rows (list): label rows, from find_label_rows
    :param px_per_degree (float): pixels per degC
    :return: ([value per row], confidence of the winning candidate), or (None, 0.0)
             if no candidate fits
    """
    # Which label slot each row is in, counted from the top row; this allows
    # for a label that was missed between two found ones
    row_ys = np.array([row["y"] for row in rows])
    slot_steps = np.round((row_ys - row_ys[0]) / (10 * px_per_degree)).astype(int)

    # Classify every glyph once, remembering which row it belongs to
    crops, owners = [], []

    for row_idx, row in enumerate(rows):
        for glyph in row["glyphs"]:
            crops.append(task2.crop_box(img, glyph, pad_frac=0.08))
            owners.append(row_idx)

    probabilities = classify_digits(crops, "print")
    per_row = [[probs for probs, owner in zip(probabilities, owners) if owner == row_idx]
               for row_idx in range(len(rows))]

    # Small constant so log() never sees zero
    eps = 1e-4
    scores = {}

    for top_value in SCALE_LABELS:
        values = [top_value - 10 * step for step in slot_steps]
        if min(values) < SCALE_LABELS[-1]:
            continue

        # Log-likelihood of this candidate across all rows
        score = 0.0
        for row, glyph_probs, value in zip(rows, per_row, values):
            text = str(abs(value))

            # Align from the right: the units digit is next to the tick
            for pos in range(1, max(len(text), len(glyph_probs)) + 1):
                if pos <= len(text) and pos <= len(glyph_probs):
                    score += math.log(glyph_probs[-pos][int(text[-pos])] + eps)
                else:
                    # A glyph missing or extra
                    score += math.log(0.02)
            if row["minus"] != (value < 0):
                # The dash is easy to miss
                score += math.log(0.25)
        scores[top_value] = score
    if not scores:
        return None, 0.0
    best_top = max(scores, key=scores.get)

    # Softmax probability of the best candidate
    total = sum(math.exp(score - scores[best_top]) for score in scores.values())
    return [best_top - 10 * step for step in slot_steps], 1.0 / total

def interpolate_reading(top_y, label_ys, label_values, ticks, px_per_degree):
    """
    interpolate_reading: Compute the temperature at row top_y

    The labels give a linear row -> degC map. Each tick near the liquid top is
    snapped to its whole degree with that map, and the reading is interpolated
    linearly between the tick just below and the tick just above the top. If
    the ticks are unusable, the label map is used directly

    :param top_y (int): row of the liquid top
    :param label_ys (list): centre row of each label
    :param label_values (list): degC value of each label
    :param ticks (list): tick rows
    :param px_per_degree (float): pixels per degC
    :return: (temperature (float), method (str), "ticks" or "labels")
    """
    # Least squares offset for the line T = offset - y / px_per_degree through the labels.
    offset = float(np.mean(np.asarray(label_values) + np.asarray(label_ys) / px_per_degree))

    def degrees(row_y):
        """
        degrees: Convert a row to degC with the current label map

        :param row_y (float): image row
        :return: temperature in degC (float)
        """
        return offset - row_y / px_per_degree

    # A label's centre is only roughly level with its tick, but the ticks sit
    # exactly on whole degrees: shift the map by the ticks' median offset from
    # the nearest whole degree
    if len(ticks):
        tick_offsets = [degrees(tick_y) - round(degrees(tick_y)) for tick_y in ticks]
        offset -= float(np.median(tick_offsets))

    tick_values = []

    for tick_y in ticks:
        temp = degrees(tick_y)
        # Ignore anything not on a tick position
        if abs(temp - round(temp)) < 0.35:
            tick_values.append((tick_y, round(temp)))

    below = [tick for tick in tick_values if tick[0] >= top_y]
    above = [tick for tick in tick_values if tick[0] < top_y]

    if below and above:
        # Nearest tick below and above the liquid top
        y_below, temp_below = min(below, key=lambda tick: tick[0])
        y_above, temp_above = max(above, key=lambda tick: tick[0])

        if temp_above > temp_below and y_below > y_above and temp_above - temp_below <= 2:
            return (temp_below + (y_below - top_y) / (y_below - y_above) * (temp_above - temp_below),
                    "ticks")
    return degrees(top_y), "labels"

def read_thermometer(img):
    """
    read_thermometer: Read the temperature (float degC) shown on a segmented scale image
    (t.png)

    :param img (np.ndarray): BGR scale image
    :return: dict with the reading "temp", the "method" used and everything found along
             the way, for printing and the debug image. Raises RuntimeError if the
             liquid or the labels cannot be found
    """
    height, width = img.shape[:2]
    response = task2.liquid_response(img)
    line = task2.fit_liquid_line(response)
    # Column of the liquid line at the centre row
    liquid_x = line[0] * height / 2 + line[1]

    mask = task2.dark_print_mask(img)
    ticks, band, tick_step = find_ticks(mask, liquid_x, height)
    # Labels lie left of the tick band; without a band, left of the tube
    band_left = band[0] if band[1] > band[0] else liquid_x - 0.1 * width
    rows = find_label_rows(img, mask, band_left, tick_step)
    if not rows:
        raise RuntimeError("no scale labels found")

    # Pixels per degC: from the ticks, else from the label spacing (10 degC)
    if tick_step:
        px_per_degree = tick_step
    elif len(rows) >= 2:
        px_per_degree = float(np.median(np.diff([row["y"] for row in rows]))) / 10
    else:
        raise RuntimeError("scale spacing not found (need ticks or two labels)")

    top = find_liquid_top(liquid_profile(response, line, width), px_per_degree)
    if top is None:
        # Fall back to Task 2's threshold method
        top = task2.find_liquid_top(response, line)
    if top is None:
        raise RuntimeError("liquid column not found")

    values, confidence = decode_labels(img, rows, px_per_degree)
    if values is None:
        raise RuntimeError("labels could not be read")
    temp, method = interpolate_reading(top, [row["y"] for row in rows], values, ticks, px_per_degree)
    return {"temp": temp, "method": method, "top": top, "line": line, "ticks": ticks,
            "band": band, "deg_px": px_per_degree, "rows": rows, "values": values,
            "label_conf": confidence}


def draw_thermo_debug(img, info):
    """
    draw_thermo_debug: Draw the liquid line, ticks, label boxes and values, and the
    reading on the scale image

    :param img (np.ndarray): BGR scale image
    :param info (dict): result of read_thermometer
    :return: annotated copy of the image (np.ndarray)
    """
    annotated = img.copy()
    height, width = img.shape[:2]

    # Line thickness and font scale relative to the image width
    thickness = max(1, width // 300)
    font_scale = max(0.4, width / 800)
    slope, intercept = info["line"]

    # Liquid line in blue
    cv2.line(annotated, (int(round(intercept)), 0), (int(round(slope * height + intercept)), height), (255, 0, 0), 1)

    # Ticks in magenta, across the tick band
    band_left, band_right = info["band"]
    for tick_y in info["ticks"]:
        cv2.line(annotated, (band_left, int(tick_y)), (band_right, int(tick_y)), (255, 0, 255), 1)

    # Label glyph boxes and decoded values in green
    for row, value in zip(info["rows"], info["values"]):
        for box_x, box_y, box_width, box_height in row["glyphs"]:
            cv2.rectangle(annotated, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 160, 0), thickness)
        # Right edge of the last glyph
        text_x = row["glyphs"][-1][0] + row["glyphs"][-1][2]
        cv2.putText(
            annotated,
            str(value),
            (text_x + 2, int(row["y"])),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 160, 0),    # green text
            thickness,
        )

    # Liquid top and the reading in red
    top = info["top"]
    cv2.line(annotated, (0, top), (width, top), (0, 0, 255), thickness)
    cv2.putText(
        annotated,
        f"{info['temp']:.1f} C",
        (width // 2 + 10, max(15, top - 5)),    # just above the liquid top, right of centre
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 255),                            # red text
        thickness,
    )

    return annotated


def round_half_up(value):
    """
    round_half_up: Round to the nearest integer, halves up (Python's round() rounds to even)

    :param value (float): number to round
    :return: the rounded integer (int)
    """
    return int(math.floor(value + 0.5))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def digit_number(path):
    """
    digit_number: Extract the number of a digit image ('d12.png' -> 12), so d10 sorts
    after d9

    :param path (str): digit image path
    :return: the number, or 0 if there is none (int)
    """
    match = re.search(r"(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else 0

def read_settings(config):
    """
    read_settings: Read the Task 3 settings from config.txt, falling back to the defaults

    :param config (dict): key/value pairs from config.txt, or None
    :return: dict with "debug"
    """
    config = config or {}
    return {"debug": int(config.get("task3_debug", DEFAULT_DEBUG)) != 0}

def process_lcd(in_dir, out_dir):
    """
    process_lcd: Recognise every digit image dN.png in in_dir and save dN.txt

    :param in_dir (str): folder of digit images from Task 2
    :param out_dir (str): folder for the text files
    :return: None
    """
    # Only d1.png, d2.png, ... (not e.g. a stray debug_*.png).
    paths = sorted((path for path in glob.glob(os.path.join(in_dir, "*.png"))
                    if re.fullmatch(r"d\d+\.png", os.path.basename(path))), key=digit_number)
    
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

    for name, (digit, confidence) in zip(names, results):
        print(f"  {name}: {digit} ({confidence:.2f})")
        save_output(os.path.join(out_dir, f"{name}.txt"), str(digit), output_type='txt')
    print(f"  bpm digits: {''.join(str(digit) for digit, _ in results)}")


def process_thermo(in_dir, out_dir, debug_path=None):
    """
    process_thermo: Read the temperature from in_dir/t.png and save it as t.txt

    :param in_dir (str): folder containing t.png from Task 2
    :param out_dir (str): folder for the text file
    :param debug_path (str): if given, save a debug image here
    :return: None
    """
    img = cv2.imread(os.path.join(in_dir, "t.png"))
    if img is None:
        print(f"  could not read {os.path.join(in_dir, 't.png')}")
        return
    
    info = read_thermometer(img)
    labels = ", ".join(f"{value} @ row {row['y']:.0f}" for value, row in zip(info["values"], info["rows"]))
    print(f"  liquid top at row {info['top']}, {len(info['ticks'])} ticks, "
          f"{info['deg_px']:.1f} px per degC")
    print(f"  labels: {labels} (confidence {info['label_conf']:.2f})")

    # The output is a whole number of degC
    reading = round_half_up(info["temp"])
    print(f"  thermo: {info['temp']:.2f} degC (from {info['method']}) -> {reading}")
    save_output(os.path.join(out_dir, "t.txt"), str(reading), output_type='txt')
    if debug_path:
        save_output(debug_path, draw_thermo_debug(img, info), output_type='image')


def run_task3(image_path, config):
    """
    run_task3: Read every segmented reading folder and save the results to output/task3/<name>/

    :param image_path (str): folder containing the lcdX/ and thermoX/ folders from Task 2
    :param config (dict): key/value pairs from config.txt
    :return: None
    """
    print("=============================================================")
    print("TASK 3 - Processing thermometer readings")
    print("=============================================================")

    settings = read_settings(config)
    # Reading folders are lcdX/ and thermoX/; anything else (e.g. a debug/
    # folder left by Task 2) is skipped.
    subdirs = sorted(folder for folder in glob.glob(os.path.join(image_path, "*"))
                     if os.path.isdir(folder) and re.match(r"(lcd|thermo)", os.path.basename(folder).lower()))
    print(f"Note: {len(subdirs)} reading(s) in {image_path} ({settings})")

    for in_dir in subdirs:
        name = os.path.basename(os.path.normpath(in_dir))
        out_dir = os.path.join(OUTPUT_DIR, name)
        print(f"[{name}]:")
        try:
            if name.lower().startswith("thermo"):
                debug_path = (os.path.join(OUTPUT_DIR, "debug", f"debug_{name}.png")
                              if settings["debug"] else None)
                process_thermo(in_dir, out_dir, debug_path)
            else:
                process_lcd(in_dir, out_dir)
            print("------")
        except Exception as error:
            # One bad reading must not stop the rest of the batch.
            print(f"Task 3 failed on {in_dir}: {error}")
