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
      condensed/expanded, with neighbouring glyphs, blur and noise.

Photos that are byte-identical to a validation image are never trained on, so
the validation digits (validation/task3) are a true held-out test set. The
script reports accuracy and a confusion matrix on synthetic hold-out data and
on those validation digits, and saves the confusion matrices as images under
runs/task3/.

Usage (inside the comp3007 conda environment):
    python train_task3.py
    python train_task3.py --epochs 20 --device cpu
"""

import argparse
import glob
import hashlib
import os

import cv2
import matplotlib
matplotlib.use("Agg")          # no interactive windows
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix
from torch import nn

import task1
import task2
import task3

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PHOTOS = os.path.join(HERE, "..", "unlabelled_data")
DEFAULT_VALIDATION = os.path.join(HERE, "..", "validation")
WEIGHTS_OUT = task3.MODEL_PATH
RUNS_DIR = os.path.join(HERE, "runs", "task3")

# Readings shown on the BPM in each capture session of the unlabelled photos
# (photo names are <date>_<time>), read off the photos by hand.
# (date, first time, last time, "SYS,DIA,PULSE")
LCD_READINGS = [
    ("20260802", "164700", "164759", "135,93,65"),
    ("20260815", "103000", "103259", "134,88,75"),
    ("20260815", "103500", "103659", "135,93,78"),
    ("20260815", "103900", "104059", "134,89,72"),
]

# Ground truth for the validation digit folders (validation/task3/lcdX),
# read off the images by hand. Used for evaluation only.
VALIDATION_LCD_LABELS = {"lcd2": "1359365", "lcd4": "1348875", "lcd5": "1348972"}

# Which segments are lit for each digit (a = top, then clockwise, g = middle).
SEGMENTS = {
    0: "abcdef", 1: "bc", 2: "abdeg", 3: "abcdg", 4: "bcfg",
    5: "acdfg", 6: "acdefg", 7: "abc", 8: "abcdefg", 9: "abcdfg",
}
# Some displays draw these digits with an extra (or one less) segment.
SEGMENT_VARIANTS = {6: "cdefg", 7: "abcf", 9: "abcfg"}

FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
FONTS = [os.path.join(FONT_DIR, f) for f in
         ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf")
         if os.path.exists(os.path.join(FONT_DIR, f))]
HERSHEY = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX]


# --------------------------------------------------------------------------- #
# Shared augmentation
# --------------------------------------------------------------------------- #

def jitter_crop(img, box, rng, pad=(-0.03, 0.06)):
    """
    Crop box (x, y, w, h) from img, moving each edge by a random fraction of
    the box height (negative = cut into the digit, positive = extra margin).
    This imitates Task 2's crops, which are tight but not perfect.
    """
    x, y, bw, bh = box
    lo = rng.uniform(pad[0], pad[1], 4) * bh
    x0, y0 = int(round(x - lo[0])), int(round(y - lo[1]))
    x1, y1 = int(round(x + bw + lo[2])), int(round(y + bh + lo[3]))
    # Pad with the edge colour so a crop past the image border still works.
    m = max(0, -x0, -y0, x1 - img.shape[1], y1 - img.shape[0])
    if m:
        img = cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_REPLICATE)
        x0, y0, x1, y1 = x0 + m, y0 + m, x1 + m, y1 + m
    x1, y1 = max(x1, x0 + 2), max(y1, y0 + 2)
    return img[y0:y1, x0:x1]


def photometric(img, rng):
    """Random blur, contrast, gamma, noise and JPEG artefacts on a grayscale crop."""
    img = img.astype(np.float32)
    h = img.shape[0]
    if rng.random() < 0.7:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, max(0.5, 0.02 * h)))
    # Contrast/brightness, then gamma.
    img = (img - img.mean()) * rng.uniform(0.6, 1.3) + img.mean() + rng.uniform(-30, 30)
    img = np.clip(img, 0, 255)
    img = 255 * (img / 255) ** rng.uniform(0.7, 1.4)
    img += rng.normal(0, rng.uniform(0, 8), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(40, 95))])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img


def small_warp(img, rng, max_angle=4.0, max_shear=0.08):
    """Small random rotation + shear about the centre, edges replicated."""
    h, w = img.shape[:2]
    a = np.radians(rng.uniform(-max_angle, max_angle))
    s = rng.uniform(-max_shear, max_shear)
    m = np.array([[np.cos(a), -np.sin(a) + s], [np.sin(a), np.cos(a)]], np.float32)
    c = np.array([w / 2, h / 2], np.float32)
    affine = np.hstack([m, (c - m @ c)[:, None]])
    return cv2.warpAffine(img, affine, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE), affine


def augment_real(img, rng):
    """Augmented copy of a real digit crop (grayscale)."""
    h = img.shape[0]
    m = int(0.1 * h) + 2
    big = cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_REPLICATE)
    big, _ = small_warp(big, rng, max_angle=3.0, max_shear=0.05)
    box = (m, m, img.shape[1], img.shape[0])
    return photometric(jitter_crop(big, box, rng, pad=(-0.02, 0.03)), rng)


# --------------------------------------------------------------------------- #
# Synthetic seven-segment digits
# --------------------------------------------------------------------------- #

def segment_polygons(w, h, t, gap):
    """Hexagonal segment outlines for a w x h digit with segment thickness t."""
    half = t / 2
    xl, xr = half, w - half
    yt, ym, yb = half, h / 2, h - half

    def horizontal(y):
        x0, x1 = xl + gap, xr - gap
        return [(x0, y), (x0 + half, y - half), (x1 - half, y - half),
                (x1, y), (x1 - half, y + half), (x0 + half, y + half)]

    def vertical(x, y0, y1):
        y0, y1 = y0 + gap, y1 - gap
        return [(x, y0), (x + half, y0 + half), (x + half, y1 - half),
                (x, y1), (x - half, y1 - half), (x - half, y0 + half)]

    return {"a": horizontal(yt), "g": horizontal(ym), "d": horizontal(yb),
            "b": vertical(xr, yt, ym), "c": vertical(xr, ym, yb),
            "f": vertical(xl, yt, ym), "e": vertical(xl, ym, yb)}


def render_lcd_digit(digit, rng):
    """One synthetic seven-segment digit crop (grayscale)."""
    lit = SEGMENTS[digit]
    if digit in SEGMENT_VARIANTS and rng.random() < 0.2:
        lit = SEGMENT_VARIANTS[digit]

    h = rng.uniform(30, 250)
    w = h * rng.uniform(0.45, 0.62)
    t = h * rng.uniform(0.08, 0.15)
    gap = t * rng.uniform(0.05, 0.35)
    m = int(0.3 * h) + 4
    size = (int(h + 2 * m), int(w + 2 * m))

    bg = rng.uniform(100, 220)
    ink = rng.uniform(20, 0.6 * bg)
    ghost = bg - rng.uniform(0.0, 0.2) * (bg - ink)   # unlit segments, faintly visible

    canvas = np.full(size, int(bg), np.uint8)
    lit_mask = np.zeros(size, np.uint8)
    shift = 4                                         # sub-pixel polygon precision
    for name, poly in segment_polygons(w, h, t, gap).items():
        pts = np.round((np.array(poly) + m) * (1 << shift)).astype(np.int32)
        colour = ink if name in lit else ghost
        cv2.fillPoly(canvas, [pts], int(colour), lineType=cv2.LINE_AA, shift=shift)
        if name in lit:
            cv2.fillPoly(lit_mask, [pts], 255, shift=shift)

    # Uneven lighting: a random brightness gradient across the digit.
    yy, xx = np.mgrid[0:size[0], 0:size[1]].astype(np.float32)
    gx, gy = rng.uniform(-0.25, 0.25, 2)
    canvas = canvas * (1 + gx * (xx / size[1] - 0.5) + gy * (yy / size[0] - 0.5))

    canvas, affine = small_warp(np.clip(canvas, 0, 255).astype(np.uint8), rng)
    lit_mask = cv2.warpAffine(lit_mask, affine, (size[1], size[0]))
    ys, xs = np.nonzero(lit_mask > 127)
    box = (xs.min(), ys.min(), xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min())
    return photometric(jitter_crop(canvas, box, rng), rng)


# --------------------------------------------------------------------------- #
# Synthetic printed digits
# --------------------------------------------------------------------------- #

def render_printed_digit(digit, rng):
    """One synthetic printed digit crop (grayscale), sometimes with neighbours."""
    size = int(rng.uniform(24, 90))
    bg = rng.uniform(140, 245)
    ink = rng.uniform(0, 0.45 * bg)
    target = str(digit)
    left = str(rng.integers(10)) if rng.random() < 0.3 else ""
    right = str(rng.integers(10)) if rng.random() < 0.3 else ""

    if FONTS and rng.random() < 0.7:
        font = ImageFont.truetype(FONTS[rng.integers(len(FONTS))], size)
        text = left + target + right
        pil = Image.new("L", (size * (len(text) + 2), size * 2), int(bg))
        draw = ImageDraw.Draw(pil)
        draw.text((size, size // 3), text, fill=int(ink), font=font)
        # Columns of the target character, from the advance widths.
        x0 = size + draw.textlength(left, font=font)
        x1 = size + draw.textlength(left + target, font=font)
        img = np.array(pil)
    else:
        face = HERSHEY[rng.integers(len(HERSHEY))]
        scale = size / 30
        thick = int(rng.integers(1, max(2, size // 8)))
        (tw, th), base = cv2.getTextSize(target, face, scale, thick)
        img = np.full((th + base + 2 * size, tw + 2 * size), bg, np.uint8)
        cv2.putText(img, target, (size, size + th), face, scale, int(ink), thick, cv2.LINE_AA)
        x0, x1 = size, size + tw

    # Condensed or expanded type.
    fx = rng.uniform(0.7, 1.15)
    img = cv2.resize(img, None, fx=fx, fy=1.0, interpolation=cv2.INTER_AREA)
    x0, x1 = int(x0 * fx), int(np.ceil(x1 * fx))
    img, affine = small_warp(img, rng, max_angle=4.0, max_shear=0.05)

    # The target glyph's box: dark pixels within its columns.
    dark = img < (bg + ink) / 2
    cols = dark[:, max(0, x0):x1]
    ys, xs = np.nonzero(cols)
    if len(xs) == 0:
        return render_printed_digit(digit, rng)
    box = (max(0, x0) + xs.min(), ys.min(), xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min())
    return photometric(jitter_crop(img, box, rng, pad=(-0.02, 0.1)), rng)


def make_synthetic(render, n, rng):
    """n rendered digits (balanced classes) as prepared network inputs + labels."""
    labels = np.arange(n) % 10
    rng.shuffle(labels)
    x = np.stack([to_uint8(task3.prepare_digit(render(int(d), rng))) for d in labels])
    return x, labels


def to_uint8(x):
    """Store prepared inputs as uint8 to keep tens of thousands in memory."""
    return np.round(x * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Real digits
# --------------------------------------------------------------------------- #

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def reading_for(photo_name):
    """The LCD reading for a photo, from its capture time, or None."""
    parts = photo_name.split("_")
    if len(parts) < 2:
        return None
    date, time = parts[0], parts[1][:6]
    for d, start, end, reading in LCD_READINGS:
        if date == d and start <= time <= end:
            return reading
    return None


def harvest_real_lcd(photo_dir, exclude_hashes):
    """
    Labelled real digit crops (grayscale) from the BPM photos.

    Each photo is run through Task 1 (LCD extraction) and Task 2 (digit
    segmentation). The crops are only kept when the row lengths match the
    known reading, so every crop gets the right label.
    """
    crops, labels = [], []
    used, skipped = 0, 0
    for path in sorted(glob.glob(os.path.join(photo_dir, "*.jpg"))):
        name = os.path.splitext(os.path.basename(path))[0]
        reading = reading_for(name)
        if reading is None:
            continue
        if file_hash(path) in exclude_hashes:
            print(f"  {name}: same photo as a validation image, held out")
            continue
        found = task1.extract_reading_region(cv2.imread(path))
        if found is None or found[0] != "bpm":
            skipped += 1
            continue
        region = found[1]
        rows = task2.find_lcd_digits(region)
        expected = reading.split(",")
        if [len(r) for r in rows] != [len(v) for v in expected]:
            skipped += 1
            continue
        for row, value in zip(rows, expected):
            for box, ch in zip(row, value):
                crop = cv2.cvtColor(task2.crop_box(region, box), cv2.COLOR_BGR2GRAY)
                crops.append(crop)
                labels.append(int(ch))
        used += 1
    print(f"  real LCD digits: {len(crops)} from {used} photo(s), {skipped} photo(s) skipped")
    return crops, np.array(labels, dtype=np.int64)


def validation_lcd(validation_dir):
    """The held-out validation digit crops and their labels, if available."""
    crops, labels = [], []
    for folder, digits in VALIDATION_LCD_LABELS.items():
        for i, ch in enumerate(digits, start=1):
            img = cv2.imread(os.path.join(validation_dir, "task3", folder, f"d{i}.png"))
            if img is not None:
                crops.append(img)
                labels.append(int(ch))
    return crops, np.array(labels, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Training and evaluation
# --------------------------------------------------------------------------- #

def to_tensor(x_uint8):
    return torch.from_numpy(x_uint8[:, None].astype(np.float32) / 255)


def predict(model, x_uint8, device, batch=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_uint8), batch):
            out.append(model(to_tensor(x_uint8[i:i + batch]).to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def train_model(x, y, x_val, y_val, args, device, name):
    """Train one DigitCNN with Adam + cosine learning-rate decay."""
    torch.manual_seed(0)
    model = task3.DigitCNN().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    rng = np.random.default_rng(0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(x))
        total = 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            xb = to_tensor(x[idx]).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        scheduler.step()
        acc = float(np.mean(predict(model, x_val, device) == y_val))
        print(f"  [{name}] epoch {epoch:2d}/{args.epochs}  loss {total / len(x):.4f}  "
              f"synthetic val acc {acc:.4f}")
    return model.cpu().eval()


def report(name, y_true, y_pred):
    """Print accuracy, per-class accuracy and the confusion matrix; save it as an image."""
    acc = float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")
    print(f"\n{name}: accuracy {acc:.4f} ({int(np.sum(y_true == y_pred))}/{len(y_true)})")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    per_class = [f"{d}: {cm[d, d]}/{cm[d].sum()}" for d in range(10) if cm[d].sum()]
    print("  per class: " + ", ".join(per_class))
    print("  confusion matrix (rows = true, columns = predicted):")
    print("     " + " ".join(f"{d:4d}" for d in range(10)))
    for d in range(10):
        print(f"  {d}: " + " ".join(f"{v:4d}" for v in cm[d]))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title(f"{name}\naccuracy {acc:.3f}")
    fig.tight_layout()
    path = os.path.join(RUNS_DIR, "confusion_" + name.lower().replace(" ", "_") + ".png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Train the Task 3 digit classifiers")
    parser.add_argument("--photos", default=DEFAULT_PHOTOS, help="unlabelled photo directory")
    parser.add_argument("--validation", default=DEFAULT_VALIDATION, help="validation set directory")
    parser.add_argument("--synthetic", type=int, default=30000, help="synthetic digits per network")
    parser.add_argument("--real-copies", type=int, default=40,
                        help="augmented copies of each real LCD digit")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RUNS_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    # Photos that are also validation images must not be trained on.
    exclude = {file_hash(p) for p in glob.glob(os.path.join(args.validation, "*", "*.jpg"))}

    print("Collecting LCD digits")
    real_crops, real_y = harvest_real_lcd(args.photos, exclude)
    real_x = [to_uint8(task3.prepare_digit(c)) for c in real_crops]
    for _ in range(args.real_copies):
        real_x += [to_uint8(task3.prepare_digit(augment_real(c, rng))) for c in real_crops]
    real_x = np.array(real_x).reshape(-1, task3.DIGIT_H, task3.DIGIT_W)
    real_y = np.tile(real_y, args.real_copies + 1)
    print(f"  {len(real_x)} real samples after augmentation")
    syn_x, syn_y = make_synthetic(render_lcd_digit, args.synthetic, rng)
    n_val = len(syn_x) // 10
    lcd_x = np.concatenate([syn_x[n_val:], real_x])
    lcd_y = np.concatenate([syn_y[n_val:], real_y])
    lcd_val_x, lcd_val_y = syn_x[:n_val], syn_y[:n_val]
    cv2.imwrite(os.path.join(RUNS_DIR, "samples_lcd.png"),
                np.vstack([np.hstack(list(lcd_x[r * 20:(r + 1) * 20])) for r in range(5)]))

    print("Collecting printed digits")
    print_x, print_y = make_synthetic(render_printed_digit, args.synthetic, rng)
    n_val = len(print_x) // 10
    cv2.imwrite(os.path.join(RUNS_DIR, "samples_print.png"),
                np.vstack([np.hstack(list(print_x[r * 20:(r + 1) * 20])) for r in range(5)]))

    print(f"Training on {device}")
    lcd_model = train_model(lcd_x, lcd_y, lcd_val_x, lcd_val_y, args, device, "lcd")
    print_model = train_model(print_x[n_val:], print_y[n_val:], print_x[:n_val], print_y[:n_val],
                              args, device, "print")

    report("LCD synthetic hold-out", lcd_val_y, predict(lcd_model, lcd_val_x, "cpu"))
    val_crops, val_y = validation_lcd(args.validation)
    if len(val_crops):
        val_x = np.array([to_uint8(task3.prepare_digit(c)) for c in val_crops])
        report("LCD validation digits", val_y, predict(lcd_model, val_x, "cpu"))
    report("Printed synthetic hold-out", print_y[:n_val],
           predict(print_model, print_x[:n_val], "cpu"))

    os.makedirs(os.path.dirname(WEIGHTS_OUT), exist_ok=True)
    torch.save({"lcd": lcd_model.state_dict(), "print": print_model.state_dict(),
                "input_size": [task3.DIGIT_H, task3.DIGIT_W]}, WEIGHTS_OUT)
    print(f"\nWeights saved to {WEIGHTS_OUT}")


if __name__ == "__main__":
    main()
