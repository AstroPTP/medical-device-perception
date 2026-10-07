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
# Last Modified: 2026-10-06

"""
Train the Task 3 digit classifiers (task3.DigitCNN), saved to data/task3_digits.pt.

Two networks with the same architecture are trained:

  - "lcd"   : seven-segment BPM digits. Training data is
      * real digits: the unlabelled BPM photos are run through Task 1 and
        Task 2, and each digit crop is labelled from the reading on the LCD.
        The photos come from four capture sessions, each showing one fixed
        reading (LCD_READINGS, read off the photos by hand), so a photo is only
        used when Task 2 found exactly the expected number of digits per row.
        Those readings never show a 0, and there are only ~200 crops, so
      * synthetic digits: seven-segment digits drawn with random segment
        shapes, unlit "ghost" segments, lighting, blur, noise, rotation and
        crop jitter, cropped to their lit segments the way Task 2 crops.
  - "print" : printed thermometer scale numbers. Synthetic only: digits drawn
      with the DejaVu fonts shipped with matplotlib and OpenCV's Hershey fonts,
      condensed/expanded, with neighbouring glyphs, blur and noise

Photos that are byte-identical to a validation image are never trained on, so
the validation digits (validation/task3) are a true held-out test set. The
script reports accuracy and a confusion matrix on synthetic hold-out data and
on those validation digits, and saves the confusion matrices as images under
runs/task3/.
"""

import argparse
import glob
import hashlib
import os

import cv2
import matplotlib
# Non-interactive backend: figures are only saved to files, no windows open
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix
from torch import nn

import task1
import task2
import task3

HERE = os.path.dirname(os.path.abspath(__file__))               # Absolute path of the directory containing this script, so paths resolve from any working directory
DEFAULT_PHOTOS = os.path.join(HERE, "..", "unlabelled_data")    # Default folder of unlabelled BPM photos, used when --photos is not given
DEFAULT_VALIDATION = os.path.join(HERE, "..", "validation")     # Default validation set folder, used when --validation is not given
WEIGHTS_OUT = task3.MODEL_PATH                                  # Checkpoint path shared with task3.py, so training and inference use the same file
RUNS_DIR = os.path.join(HERE, "runs", "task3")                  # Output folder for the sample grids and confusion matrix images

# Readings shown on the BPM in each capture session of the unlabelled photos
# (photo names are <date>_<time>), read off the photos by hand
# (date, first time, last time, "SYS,DIA,PULSE")
LCD_READINGS = [
    ("20260802", "164700", "164759", "135,93,65"),
    ("20260815", "103000", "103259", "134,88,75"),
    ("20260815", "103500", "103659", "135,93,78"),
    ("20260815", "103900", "104059", "134,89,72"),
]

# Ground truth for the validation digit folders (validation/task3/lcdX),
# read off the images by hand. Used for evaluation only
VALIDATION_LCD_LABELS = {"lcd2": "1359365", "lcd4": "1348875", "lcd5": "1348972"}

# Which segments are lit for each digit (a = top, then clockwise, g = middle)
SEGMENTS = {
    0: "abcdef", 1: "bc", 2: "abdeg", 3: "abcdg", 4: "bcfg",
    5: "acdfg", 6: "acdefg", 7: "abc", 8: "abcdefg", 9: "abcdfg",
}
# Some displays draw these digits with an extra (or one less) segment
SEGMENT_VARIANTS = {6: "cdefg", 7: "abcf", 9: "abcfg"}

# TrueType fonts shipped with matplotlib, so no system fonts are needed
FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
# DejaVu fonts for printed digits, keeping only those present in this matplotlib install
FONTS = [os.path.join(FONT_DIR, font_file) for font_file in
         ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf")
         if os.path.exists(os.path.join(FONT_DIR, font_file))]
# OpenCV built-in fonts, used as an alternative printed style
HERSHEY = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX]

# --------------------------------------------------------------------------- #
# Shared augmentation
# --------------------------------------------------------------------------- #

def jitter_crop(img, box, rand_gen, padding=(-0.03, 0.06)):
    """
    jitter_crop: Crop box (x, y, w, h) from img, moving each edge by a random fraction
    of the box height (negative = cut into the digit, positive = extra margin).
    This imitates Task 2's crops, which are tight but not perfect

    :param img (np.ndarray): grayscale source image
    :param box (tuple): digit box as (x, y, w, h) in pixels
    :param rand_gen (np.random.Generator): random generator for the edge offsets
    :param padding (tuple): (min, max) edge offset as a fraction of the box height
    :return: cropped image region (np.ndarray)
    """
    box_x, box_y, box_width, box_height = box
    # Random offset for each edge (left, top, right, bottom), scaled by the box height
    edge_offsets = rand_gen.uniform(padding[0], padding[1], 4) * box_height
    left, top = int(round(box_x - edge_offsets[0])), int(round(box_y - edge_offsets[1]))
    right = int(round(box_x + box_width + edge_offsets[2]))
    bottom = int(round(box_y + box_height + edge_offsets[3]))

    # padding with the edge colour so a crop past the image border still works
    margin = max(0, -left, -top, right - img.shape[1], bottom - img.shape[0])
    if margin:
        img = cv2.copyMakeBorder(
            img,
            margin,                 # top margin
            margin,                 # bottom margin
            margin,                 # left margin
            margin,                 # right margin
            cv2.BORDER_REPLICATE,   # repeat the edge pixels
        )
        # Shift the crop corners into the padded image
        left, top, right, bottom = left + margin, top + margin, right + margin, bottom + margin
    # Keep the crop at least 2 px wide and tall
    right, bottom = max(right, left + 2), max(bottom, top + 2)

    return img[top:bottom, left:right]

def photometric(img, rand_gen):
    """
    photometric: Apply random blur, contrast, gamma, noise and JPEG artefacts to a
    grayscale crop

    :param img (np.ndarray): grayscale digit crop (uint8)
    :param rand_gen (np.random.Generator): random generator for the augmentation strengths
    :return: augmented crop (np.ndarray, uint8)
    """
    # Work in float so the adjustments below do not overflow uint8
    img = img.astype(np.float32)
    height = img.shape[0]

    # Gaussian blur on 70% of crops, with the strength scaled by the crop height
    if rand_gen.random() < 0.7:
        img = cv2.GaussianBlur(img, (0, 0), rand_gen.uniform(0.3, max(0.5, 0.02 * height)))
    
    # Contrast/brightness, then gamma
    img = (img - img.mean()) * rand_gen.uniform(0.6, 1.3) + img.mean() + rand_gen.uniform(-30, 30)
    img = np.clip(img, 0, 255)
    img = 255 * (img / 255) ** rand_gen.uniform(0.7, 1.4)

    # Additive Gaussian sensor noise
    img += rand_gen.normal(0, rand_gen.uniform(0, 8), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # JPEG compression artefacts on half of the crops
    if rand_gen.random() < 0.5:
        jpeg_quality = int(rand_gen.uniform(40, 95))
        success, jpeg_buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        img = cv2.imdecode(jpeg_buffer, cv2.IMREAD_GRAYSCALE)
    
    return img

def small_warp(img, rand_gen, max_angle=4.0, max_shear=0.08):
    """
    small_warp: Apply a small random rotation and shear about the image centre, with
    the edges replicated

    :param img (np.ndarray): image to warp
    :param rand_gen (np.random.Generator): random generator for the angle and shear
    :param max_angle (float): maximum rotation in degrees, either direction
    :param max_shear (float): maximum horizontal shear factor, either direction
    :return: warped image (np.ndarray) and the 2x3 affine matrix used (np.ndarray)
    """
    height, width = img.shape[:2]
    # Random rotation angle (radians) and horizontal shear
    angle = np.radians(rand_gen.uniform(-max_angle, max_angle))
    shear = rand_gen.uniform(-max_shear, max_shear)

    # 2x2 rotation matrix with the shear added to the top-right term
    rotation_shear = np.array([[np.cos(angle), -np.sin(angle) + shear],
                               [np.sin(angle), np.cos(angle)]], np.float32)
    
    # Translation that keeps the image centre fixed
    centre = np.array([width / 2, height / 2], np.float32)
    affine = np.hstack([rotation_shear, (centre - rotation_shear @ centre)[:, None]])

    return cv2.warpAffine(
        img,
        affine,
        (width, height),                    # output size matches the input
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,    # fill exposed corners with edge pixels
    ), affine


def augment_real(img, rand_gen):
    """
    augment_real: Create an augmented copy of a real digit crop

    :param img (np.ndarray): grayscale real digit crop
    :param rand_gen (np.random.Generator): random generator for the augmentations
    :return: augmented crop (np.ndarray, uint8)
    """
    height = img.shape[0]
    # Margin of 10% of the height, so the warp does not pull in empty corners
    margin = int(0.1 * height) + 2
    padded = cv2.copyMakeBorder(
        img,
        margin,                 # top margin
        margin,                 # bottom margin
        margin,                 # left margin
        margin,                 # right margin
        cv2.BORDER_REPLICATE,   # repeat the edge pixels
    )
    padded, _ = small_warp(
        padded,
        rand_gen,
        max_angle=3.0,      # milder than the synthetic digits, since real crops already vary
        max_shear=0.05,
    )
    # Original digit box inside the padded image
    box = (margin, margin, img.shape[1], img.shape[0])

    # Tight crop jitter, then photometric noise
    return photometric(jitter_crop(padded, box, rand_gen, padding=(-0.02, 0.03)), rand_gen)


# --------------------------------------------------------------------------- #
# Synthetic seven-segment digits
# --------------------------------------------------------------------------- #

def segment_polygons(width, height, thickness, gap):
    """
    segment_polygons: Build hexagonal segment outlines for a width x height
    seven-segment digit

    :param width (float): digit width in pixels
    :param height (float): digit height in pixels
    :param thickness (float): segment thickness in pixels
    :param gap (float): space left between neighbouring segments in pixels
    :return: dict mapping each segment name ("a" to "g") to its list of (x, y) vertices
    """
    half_thick = thickness / 2
    # Segment centre lines: left/right x and top/middle/bottom y, inset by half a thickness
    x_left, x_right = half_thick, width - half_thick
    y_top, y_middle, y_bottom = half_thick, height / 2, height - half_thick

    def horizontal(y):
        """
        horizontal: Build a horizontal hexagonal segment centred on row y

        :param y (float): y coordinate of the segment centre line
        :return: list of six (x, y) vertices
        """
        x_start, x_end = x_left + gap, x_right - gap

        return [(x_start, y), (x_start + half_thick, y - half_thick),
                (x_end - half_thick, y - half_thick), (x_end, y),
                (x_end - half_thick, y + half_thick), (x_start + half_thick, y + half_thick)]

    def vertical(x, y_start, y_end):
        """
        vertical: Build a vertical hexagonal segment on column x between rows y_start
        and y_end

        :param x (float): x coordinate of the segment centre line
        :param y_start (float): top end of the segment
        :param y_end (float): bottom end of the segment
        :return: list of six (x, y) vertices
        """
        y_start, y_end = y_start + gap, y_end - gap

        return [(x, y_start), (x + half_thick, y_start + half_thick),
                (x + half_thick, y_end - half_thick), (x, y_end),
                (x - half_thick, y_end - half_thick), (x - half_thick, y_start + half_thick)]

    # a/g/d are the top/middle/bottom bars; b/c the right side, f/e the left side
    return {"a": horizontal(y_top), "g": horizontal(y_middle), "d": horizontal(y_bottom),
            "b": vertical(x_right, y_top, y_middle), "c": vertical(x_right, y_middle, y_bottom),
            "f": vertical(x_left, y_top, y_middle), "e": vertical(x_left, y_middle, y_bottom)}


def render_lcd_digit(digit, rand_gen):
    """
    render_lcd_digit: Render one synthetic seven-segment digit crop

    :param digit (int): digit to draw, 0 to 9
    :param rand_gen (np.random.Generator): random generator for shape, colour and augmentation
    :return: grayscale digit crop (np.ndarray, uint8)
    """
    lit_segments = SEGMENTS[digit]
    # Use the alternative segment set for 6, 7 or 9 on 20% of renders
    if digit in SEGMENT_VARIANTS and rand_gen.random() < 0.2:
        lit_segments = SEGMENT_VARIANTS[digit]

    # Random digit height, width ratio, segment thickness and gap between segments
    height = rand_gen.uniform(30, 250)
    width = height * rand_gen.uniform(0.45, 0.62)
    thickness = height * rand_gen.uniform(0.08, 0.15)
    gap = thickness * rand_gen.uniform(0.05, 0.35)

    # Margin around the digit so the warp and crop jitter stay inside the canvas
    margin = int(0.3 * height) + 4
    canvas_size = (int(height + 2 * margin), int(width + 2 * margin))

    # Background and lit-segment grey levels
    background = rand_gen.uniform(100, 220)
    ink = rand_gen.uniform(20, 0.6 * background)

    # Unlit segments, slightly darker than the background so they are faintly visible
    ghost = background - rand_gen.uniform(0.0, 0.2) * (background - ink)

    canvas = np.full(canvas_size, int(background), np.uint8)
    # Mask of the lit segments, used later to find the digit box
    lit_mask = np.zeros(canvas_size, np.uint8)
    # Polygon coordinates use 4 fractional bits for sub-pixel precision
    shift = 4

    # Draw every segment: lit ones in ink, unlit ones as ghosts
    for segment_name, polygon in segment_polygons(width, height, thickness, gap).items():
        points = np.round((np.array(polygon) + margin) * (1 << shift)).astype(np.int32)
        colour = ink if segment_name in lit_segments else ghost
        cv2.fillPoly(canvas, [points], int(colour), lineType=cv2.LINE_AA, shift=shift)
        if segment_name in lit_segments:
            cv2.fillPoly(lit_mask, [points], 255, shift=shift)

    # Uneven lighting: a random brightness gradient across the digit
    grid_y, grid_x = np.mgrid[0:canvas_size[0], 0:canvas_size[1]].astype(np.float32)
    gradient_x, gradient_y = rand_gen.uniform(-0.25, 0.25, 2)
    canvas = canvas * (1 + gradient_x * (grid_x / canvas_size[1] - 0.5)
                       + gradient_y * (grid_y / canvas_size[0] - 0.5))

    # Warp the digit and its mask with the same affine so they stay aligned
    canvas, affine = small_warp(np.clip(canvas, 0, 255).astype(np.uint8), rand_gen)
    lit_mask = cv2.warpAffine(lit_mask, affine, (canvas_size[1], canvas_size[0]))

    # Digit box = bounding box of the lit segments, matching how Task 2 crops
    lit_rows, lit_cols = np.nonzero(lit_mask > 127)
    box = (lit_cols.min(), lit_rows.min(),
           lit_cols.max() + 1 - lit_cols.min(), lit_rows.max() + 1 - lit_rows.min())
    
    return photometric(jitter_crop(canvas, box, rand_gen), rand_gen)


# --------------------------------------------------------------------------- #
# Synthetic printed digits
# --------------------------------------------------------------------------- #

def render_printed_digit(digit, rand_gen):
    """
    render_printed_digit: Render one synthetic printed digit crop, sometimes with
    neighbouring digits partly visible at the sides

    :param digit (int): digit to draw, 0 to 9
    :param rand_gen (np.random.Generator): random generator for font, colour and augmentation
    :return: grayscale digit crop (np.ndarray, uint8)
    """
    # Font size, background and ink grey levels
    font_size = int(rand_gen.uniform(24, 90))
    background = rand_gen.uniform(140, 245)
    ink = rand_gen.uniform(0, 0.45 * background)
    target_char = str(digit)

    # Optional neighbouring digit on each side (30% chance each), as on a printed scale
    left_neighbour = str(rand_gen.integers(10)) if rand_gen.random() < 0.3 else ""
    right_neighbour = str(rand_gen.integers(10)) if rand_gen.random() < 0.3 else ""

    # DejaVu TrueType font on 70% of digits, otherwise an OpenCV Hershey font
    if FONTS and rand_gen.random() < 0.7:
        font = ImageFont.truetype(FONTS[rand_gen.integers(len(FONTS))], font_size)
        text = left_neighbour + target_char + right_neighbour

        # Canvas with a one-character margin on each side
        pil_image = Image.new("L", (font_size * (len(text) + 2), font_size * 2), int(background))
        draw = ImageDraw.Draw(pil_image)
        draw.text((font_size, font_size // 3), text, fill=int(ink), font=font)

        # Columns of the target character, from the advance widths
        glyph_left = font_size + draw.textlength(left_neighbour, font=font)
        glyph_right = font_size + draw.textlength(left_neighbour + target_char, font=font)
        img = np.array(pil_image)
    else:
        font_face = HERSHEY[rand_gen.integers(len(HERSHEY))]
        # Hershey font scale and stroke thickness derived from the font size
        font_scale = font_size / 30
        thickness = int(rand_gen.integers(1, max(2, font_size // 8)))
        (text_width, text_height), baseline = cv2.getTextSize(target_char, font_face, font_scale, thickness)

        # Canvas with a margin of one font size on each side
        img = np.full((text_height + baseline + 2 * font_size, text_width + 2 * font_size),
                      background, np.uint8)
        cv2.putText(
            img,
            target_char,
            (font_size, font_size + text_height),   # bottom-left corner of the text
            font_face,
            font_scale,
            int(ink),                               # text colour
            thickness,
            cv2.LINE_AA,                            # anti-aliased strokes
        )

        # Column range of the target glyph
        glyph_left, glyph_right = font_size, font_size + text_width

    # Condensed or expanded type
    stretch_x = rand_gen.uniform(0.7, 1.15)
    img = cv2.resize(img, None, fx=stretch_x, fy=1.0, interpolation=cv2.INTER_AREA)

    # Scale the target glyph's columns by the same factor
    glyph_left, glyph_right = int(glyph_left * stretch_x), int(np.ceil(glyph_right * stretch_x))
    img, affine = small_warp(img, rand_gen, max_angle=4.0, max_shear=0.05)

    # The target glyph's box: dark pixels within its columns
    dark_mask = img < (background + ink) / 2
    glyph_columns = dark_mask[:, max(0, glyph_left):glyph_right]
    dark_rows, dark_cols = np.nonzero(glyph_columns)

    # No dark pixels left in the target columns: render a new digit instead
    if len(dark_cols) == 0:
        return render_printed_digit(digit, rand_gen)
    box = (max(0, glyph_left) + dark_cols.min(), dark_rows.min(),
           dark_cols.max() + 1 - dark_cols.min(), dark_rows.max() + 1 - dark_rows.min())
    
    return photometric(jitter_crop(img, box, rand_gen, padding=(-0.02, 0.1)), rand_gen)


def make_synthetic(render, count, rand_gen):
    """
    make_synthetic: Render count digits with balanced classes and prepare them as
    network inputs

    :param render (callable): render function taking (digit, rand_gen) and returning a crop
    :param count (int): number of digits to render
    :param rand_gen (np.random.Generator): random generator passed to the render function
    :return: prepared inputs (np.ndarray, uint8) and their labels (np.ndarray)
    """
    # Equal count of each digit 0-9, in random order
    labels = np.arange(count) % 10
    rand_gen.shuffle(labels)
    # Render each digit, resize it to the network input and store it as uint8
    inputs = np.stack([to_uint8(task3.prepare_digit(render(int(digit), rand_gen)))
                       for digit in labels])
    
    return inputs, labels


def to_uint8(values):
    """
    to_uint8: Convert prepared inputs in [0, 1] to uint8 to keep tens of thousands in
    memory

    :param values (np.ndarray): prepared input with values in [0, 1]
    :return: the same input scaled to 0-255 (np.ndarray, uint8)
    """
    return np.round(values * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Real digits
# --------------------------------------------------------------------------- #

def file_hash(path):
    """
    file_hash: Compute the MD5 hash of a file's bytes, used to detect identical photos

    :param path (str): path to the file
    :return: hexadecimal MD5 digest (str)
    """
    with open(path, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()


def reading_for(photo_name):
    """
    reading_for: Look up the LCD reading for a photo from its capture time

    :param photo_name (str): photo file name without extension, formatted <date>_<time>
    :return: reading as "SYS,DIA,PULSE" (str), or None when no capture session matches
    """
    # Names without both a date and a time part cannot be matched
    parts = photo_name.split("_")
    if len(parts) < 2:
        return None
    # The first 6 characters of the time part are HHMMSS
    date, time = parts[0], parts[1][:6]
    # Find the capture session whose time range contains this photo
    for session_date, start, end, reading in LCD_READINGS:
        if date == session_date and start <= time <= end:
            return reading
    
    return None


def harvest_real_lcd(photo_dir, exclude_hashes):
    """
    harvest_real_lcd: Collect labelled real digit crops from the BPM photos

    Each photo is run through Task 1 (LCD extraction) and Task 2 (digit
    segmentation). The crops are only kept when the row lengths match the
    known reading, so every crop gets the right label

    :param photo_dir (str): folder containing the unlabelled .jpg photos
    :param exclude_hashes (set): MD5 hashes of photos that must not be used
    :return: list of grayscale digit crops and their labels (np.ndarray, int64)
    """
    crops, labels = [], []
    used, skipped = 0, 0

    for path in sorted(glob.glob(os.path.join(photo_dir, "*.jpg"))):
        name = os.path.splitext(os.path.basename(path))[0]
        # Photos outside the known capture sessions have no label; only labelled
        # photos are used
        reading = reading_for(name)
        if reading is not None:
            # Hold out photos that are byte-identical to a validation image
            if file_hash(path) in exclude_hashes:
                print(f"  {name}: same photo as a validation image, held out")
            else:
                # Task 1: locate the device and extract its reading region; only BPM photos are used
                detection = task1.extract_reading_region(cv2.imread(path))
                if detection is None or detection[0] != "bpm":
                    skipped += 1
                else:
                    region = detection[1]

                    # Task 2: split the LCD into rows of digit boxes
                    rows = task2.find_lcd_digits(region)
                    expected = reading.split(",")

                    # Keep the photo only when every row has as many digits as the expected reading
                    if [len(row) for row in rows] != [len(value) for value in expected]:
                        skipped += 1
                    else:
                        # Label each digit crop with the matching character of the reading
                        for row, value in zip(rows, expected):
                            for box, char in zip(row, value):
                                crop = cv2.cvtColor(task2.crop_box(region, box), cv2.COLOR_BGR2GRAY)
                                crops.append(crop)
                                labels.append(int(char))
                        used += 1
    print(f"  real LCD digits: {len(crops)} from {used} photo(s), {skipped} photo(s) skipped")

    return crops, np.array(labels, dtype=np.int64)


def validation_lcd(validation_dir):
    """
    validation_lcd: Load the held-out validation digit crops and their labels, if
    available

    :param validation_dir (str): validation set folder containing task3/lcdX subfolders
    :return: list of digit crops (BGR) and their labels (np.ndarray, int64)
    """
    crops, labels = [], []
    for folder, digits in VALIDATION_LCD_LABELS.items():
        # Crops are named d1.png, d2.png, ... in reading order; missing files are skipped
        for position, char in enumerate(digits, start=1):
            img = cv2.imread(os.path.join(validation_dir, "task3", folder, f"d{position}.png"))
            if img is not None:
                crops.append(img)
                labels.append(int(char))

    return crops, np.array(labels, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #

def to_tensor(inputs_uint8):
    """
    to_tensor: Convert a batch of uint8 inputs to a float tensor in [0, 1] with a
    channel axis

    :param inputs_uint8 (np.ndarray): inputs of shape (N, H, W), uint8
    :return: tensor of shape (N, 1, H, W), float32 (torch.Tensor)
    """
    return torch.from_numpy(inputs_uint8[:, None].astype(np.float32) / 255)


def predict(model, inputs_uint8, device, batch_size=512):
    """
    predict: Predict the digit class for every input, in batches

    :param model (task3.DigitCNN): trained network
    :param inputs_uint8 (np.ndarray): inputs of shape (N, H, W), uint8
    :param device (str): device to run the network on, "cuda" or "cpu"
    :param batch_size (int): number of inputs per forward pass
    :return: predicted digit for each input (np.ndarray)
    """
    # Evaluation mode and no gradient tracking for inference
    model.eval()
    predictions = []
    with torch.no_grad():
        # Keep the highest-scoring class of each input
        for start in range(0, len(inputs_uint8), batch_size):
            batch = to_tensor(inputs_uint8[start:start + batch_size]).to(device)
            predictions.append(model(batch).argmax(1).cpu().numpy())

    # Empty input gives an empty prediction array
    return np.concatenate(predictions) if predictions else np.zeros(0, np.int64)


def train_model(train_inputs, train_labels, val_inputs, val_labels, args, device, name):
    """
    train_model: Train one DigitCNN with Adam and cosine learning-rate decay

    :param train_inputs (np.ndarray): training inputs of shape (N, H, W), uint8
    :param train_labels (np.ndarray): training labels
    :param val_inputs (np.ndarray): hold-out inputs, used to report accuracy each epoch
    :param val_labels (np.ndarray): hold-out labels
    :param args (argparse.Namespace): parsed options (epochs, batch, lr)
    :param device (str): training device, "cuda" or "cpu"
    :param name (str): network name shown in the training log
    :return: trained network on the CPU in evaluation mode (task3.DigitCNN)
    """
    # Fixed seed so the initial weights are reproducible
    torch.manual_seed(0)
    model = task3.DigitCNN().to(device)
    
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)         # Adam with light L2 weight decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)    # Cosine decay of the learning rate over all epochs  
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)                                     # Cross-entropy with mild label smoothing to avoid over-confident predictions
    rand_gen = np.random.default_rng(0)                                                     # Separate generator for shuffling the training order

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Shuffle the sample order every epoch
        order = rand_gen.permutation(len(train_inputs))
        total_loss = 0.0

        for start in range(0, len(order), args.batch):
            batch_idx = order[start:start + args.batch]
            batch_inputs = to_tensor(train_inputs[batch_idx]).to(device)
            batch_labels = torch.from_numpy(train_labels[batch_idx]).to(device)

            # One mini-batch step: forward pass, loss, backward pass, weight update
            optimiser.zero_grad()
            loss = loss_fn(model(batch_inputs), batch_labels)
            loss.backward()
            optimiser.step()

            # Accumulate the loss weighted by batch size for the epoch average
            total_loss += loss.item() * len(batch_idx)

        # Step the learning-rate schedule once per epoch
        scheduler.step()
        # Accuracy on the synthetic hold-out set
        accuracy = float(np.mean(predict(model, val_inputs, device) == val_labels))
        print(f"  [{name}] epoch {epoch:2d}/{args.epochs}  loss {total_loss / len(train_inputs):.4f}  "
              f"synthetic val acc {accuracy:.4f}")
        
    # Move to the CPU in evaluation mode, ready for evaluation and saving
    return model.cpu().eval()


def report(name, y_true, y_pred):
    """
    report: Print accuracy, per-class accuracy and the confusion matrix, and save the
    confusion matrix as an image under RUNS_DIR

    :param name (str): report title, also used to build the image file name
    :param y_true (np.ndarray): true digit labels
    :param y_pred (np.ndarray): predicted digit labels
    :return: None
    """
    # Overall accuracy; NaN when there are no samples
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")
    print(f"\n{name}: accuracy {accuracy:.4f} ({int(np.sum(y_true == y_pred))}/{len(y_true)})")

    # Confusion matrix over all 10 digits (rows = true, columns = predicted)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    # Correct/total for each digit that appears in y_true
    per_class = [f"{digit}: {conf_matrix[digit, digit]}/{conf_matrix[digit].sum()}"
                 for digit in range(10) if conf_matrix[digit].sum()]
    print("  per class: " + ", ".join(per_class))
    # Text version of the confusion matrix
    print("  confusion matrix (rows = true, columns = predicted):")
    print("     " + " ".join(f"{digit:4d}" for digit in range(10)))
    for digit in range(10):
        print(f"  {digit}: " + " ".join(f"{cell:4d}" for cell in conf_matrix[digit]))

    # Heatmap version of the confusion matrix, with counts written in non-empty cells
    figure, axes = plt.subplots(figsize=(5, 4.5))
    axes.imshow(conf_matrix, cmap="Blues")
    for true_digit in range(10):
        for pred_digit in range(10):
            cell = conf_matrix[true_digit, pred_digit]
            if cell:
                axes.text(
                    pred_digit,
                    true_digit,
                    str(cell),
                    ha="center",
                    va="center",
                    color="white" if cell > conf_matrix.max() / 2 else "black",   # white text on dark cells
                    fontsize=8,
                )
    axes.set_xticks(range(10))
    axes.set_yticks(range(10))
    axes.set_xlabel("Predicted digit")
    axes.set_ylabel("True digit")
    axes.set_title(f"{name}\naccuracy {accuracy:.3f}")
    figure.tight_layout()

    # File name built from the report name, e.g. confusion_lcd_validation_digits.png
    path = os.path.join(RUNS_DIR, "confusion_" + name.lower().replace(" ", "_") + ".png")
    figure.savefig(path, dpi=120)
    plt.close(figure)
    print(f"  saved {path}")


def main():
    """
    main: Parse command-line arguments, build the real and synthetic digit sets, train
    the LCD and printed-digit networks, report their accuracy and save both to
    data/task3_digits.pt

    :return: None
    """
    # Command-line options; every option has a default, so the script runs with no arguments
    parser = argparse.ArgumentParser(description="Train the Task 3 digit classifiers")
    # Folder of unlabelled BPM photos used for the real LCD digits
    parser.add_argument("--photos", default=DEFAULT_PHOTOS, help="unlabelled photo directory")
    # Validation set, held out from training and used for the final report
    parser.add_argument("--validation", default=DEFAULT_VALIDATION, help="validation set directory")
    # Number of synthetic digits rendered for each network
    parser.add_argument("--synthetic", type=int, default=30000, help="synthetic digits per network")
    parser.add_argument(
        "--real-copies",
        type=int,
        default=40,         # ~200 real crops become ~8,000 training samples
        help="augmented copies of each real LCD digit",
    )
    # Number of passes over the training set
    parser.add_argument("--epochs", type=int, default=15)
    # Samples per mini-batch
    parser.add_argument("--batch", type=int, default=128)
    # Initial Adam learning rate, decayed by the cosine schedule
    parser.add_argument("--lr", type=float, default=0.001)
    # Training device; None picks the GPU if one is available
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = parser.parse_args()

    # Use the GPU when available unless --device is given
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RUNS_DIR, exist_ok=True)
    # Fixed seed for reproducible data generation and augmentation
    rand_gen = np.random.default_rng(0)

    # Photos that are also validation images must not be trained on.
    exclude_hashes = {file_hash(photo_path)
                      for photo_path in glob.glob(os.path.join(args.validation, "*", "*.jpg"))}

    print("Collecting LCD digits")
    real_crops, real_labels = harvest_real_lcd(args.photos, exclude_hashes)
    # Each real crop once as is, then args.real_copies augmented copies of every crop
    real_inputs = [to_uint8(task3.prepare_digit(crop)) for crop in real_crops]

    for _ in range(args.real_copies):
        real_inputs += [to_uint8(task3.prepare_digit(augment_real(crop, rand_gen)))
                        for crop in real_crops]
    real_inputs = np.array(real_inputs).reshape(-1, task3.DIGIT_H, task3.DIGIT_W)
    # Repeat the labels once per copy, matching the order of real_inputs
    real_labels = np.tile(real_labels, args.real_copies + 1)
    print(f"  {len(real_inputs)} real samples after augmentation")

    # Synthetic seven-segment digits; the first 10% is held out for validation
    syn_inputs, syn_labels = make_synthetic(render_lcd_digit, args.synthetic, rand_gen)
    val_count = len(syn_inputs) // 10

    # Training set: the remaining synthetic digits plus all real samples
    lcd_inputs = np.concatenate([syn_inputs[val_count:], real_inputs])
    lcd_labels = np.concatenate([syn_labels[val_count:], real_labels])
    lcd_val_inputs, lcd_val_labels = syn_inputs[:val_count], syn_labels[:val_count]
    
    # Save a 5 x 20 grid of training samples for visual inspection
    cv2.imwrite(os.path.join(RUNS_DIR, "samples_lcd.png"),
                np.vstack([np.hstack(list(lcd_inputs[row * 20:(row + 1) * 20])) for row in range(5)]))

    print("Collecting printed digits")
    # Synthetic printed digits; the first 10% is held out for validation
    print_inputs, print_labels = make_synthetic(render_printed_digit, args.synthetic, rand_gen)
    val_count = len(print_inputs) // 10
    # Save a 5 x 20 grid of samples for visual inspection
    cv2.imwrite(os.path.join(RUNS_DIR, "samples_print.png"),
                np.vstack([np.hstack(list(print_inputs[row * 20:(row + 1) * 20])) for row in range(5)]))

    print(f"Training on {device}")
    lcd_model = train_model(
        lcd_inputs,
        lcd_labels,
        lcd_val_inputs,
        lcd_val_labels,
        args,
        device,
        "lcd",                          # name shown in the training log
    )
    print_model = train_model(
        print_inputs[val_count:],       # training split
        print_labels[val_count:],
        print_inputs[:val_count],       # synthetic hold-out split
        print_labels[:val_count],
        args,
        device,
        "print",                        # name shown in the training log
    )

    # Evaluate on the CPU, where train_model returns the networks
    report("LCD synthetic hold-out", lcd_val_labels, predict(lcd_model, lcd_val_inputs, "cpu"))
    # Real validation digits, only when the validation folder is available
    val_crops, val_labels = validation_lcd(args.validation)
    if len(val_crops):
        val_inputs = np.array([to_uint8(task3.prepare_digit(crop)) for crop in val_crops])
        report("LCD validation digits", val_labels, predict(lcd_model, val_inputs, "cpu"))
    report("Printed synthetic hold-out", print_labels[:val_count],
           predict(print_model, print_inputs[:val_count], "cpu"))

    # Save both networks and the input size in one checkpoint, which task3.py loads
    os.makedirs(os.path.dirname(WEIGHTS_OUT), exist_ok=True)
    torch.save({"lcd": lcd_model.state_dict(), "print": print_model.state_dict(),
                "input_size": [task3.DIGIT_H, task3.DIGIT_W]}, WEIGHTS_OUT)
    print(f"\nWeights saved to {WEIGHTS_OUT}")


if __name__ == "__main__":
    main()
