

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
# Last Modified: 2026-10-04

"""
Task 1: detect the medical device in each photo and extract its reading region.

A YOLOv8n-pose model (trained by train_task1.py, weights in data/task1_pose.pt)
predicts, for each device, its class (bpm or thermo) and the four corners of
the reading region in the device's own reading orientation:
0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left.

Those corners give a perspective transform that maps the region to an upright
rectangle, which handles rotation and perspective distortion in one step.

  - BPM         -> output/task1/lcdX.png     (the LCD panel)
  - Thermometer -> output/task1/thermoX.png  (the thermometer plate, upright)
  - No device   -> no output file
"""

import glob
import os
import re

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "data", "task1_pose.pt")
OUTPUT_DIR = os.path.join(HERE, "output", "task1")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Defaults, overridable from config.txt (e.g. "task1_conf: 0.6").
DEFAULT_CONF = 0.5     # minimum detection confidence; below this = negative image
DEFAULT_IMGSZ = 960    # inference size, same as training
DEFAULT_PAD = 0.0      # grow the corner quad outwards by this fraction of its size

# Output file prefix for each class name the model predicts.
PREFIX = {"bpm": "lcd", "thermo": "thermo"}

_model = None


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
# Detection
# --------------------------------------------------------------------------- #

def load_model():
    """Load the pose model once and reuse it (Task 4 calls this per image)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model weights not found: {MODEL_PATH} (run train_task1.py)")
        # Imported here so tasks that never detect do not pay the import cost.
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
        print(f"Loaded detector: {MODEL_PATH}")
    return _model


def detect_device(img, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ):
    """
    Return (class_name, confidence, corners) for the most confident device,
    or None if nothing scores above `conf`.

    Each positive image holds exactly one device, so only the best detection
    is kept; extra boxes would otherwise count as failed detections.
    corners is a (4, 2) float array in pixel coordinates: TL, TR, BR, BL.
    """
    model = load_model()
    result = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best = int(result.boxes.conf.argmax())
    class_name = result.names[int(result.boxes.cls[best])]
    confidence = float(result.boxes.conf[best])
    corners = result.keypoints.xy[best].cpu().numpy().astype(np.float32)
    return class_name, confidence, corners


# --------------------------------------------------------------------------- #
# Geometric correction
# --------------------------------------------------------------------------- #

def ensure_clockwise(corners):
    """
    Make sure the corners go round clockwise (in image coordinates).

    The model is trained on clockwise TL, TR, BR, BL labels, so this is only a
    safety net: an anticlockwise quad would produce a mirrored output. Keeping
    corner 0 and reversing the direction preserves which side is "top".
    """
    x, y = corners[:, 0], corners[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if signed_area < 0:
        corners = corners[[0, 3, 2, 1]]
    return corners


def expand_quad(corners, pad_frac):
    """Push each corner away from the centre by pad_frac (0 = unchanged)."""
    if pad_frac <= 0:
        return corners
    centre = corners.mean(axis=0)
    return centre + (corners - centre) * (1.0 + pad_frac)


def rectify(img, corners):
    """
    Warp the quad TL, TR, BR, BL to an upright rectangle.

    The output size is the longer of each pair of opposite edges, so the
    region keeps (roughly) its original resolution and aspect ratio.
    """
    tl, tr, br, bl = corners
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if width < 2 or height < 2:
        raise ValueError(f"degenerate reading region ({width}x{height})")

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    # Replicate the border so corners slightly outside the photo do not
    # leave black wedges in the crop.
    return cv2.warpPerspective(img, H, (width, height), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def extract_reading_region(img, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ, pad=DEFAULT_PAD):
    """
    Full Task 1 step for one image.

    Returns (class_name, region_image, confidence), or None for a negative image.
    Shared with Task 4.
    """
    detection = detect_device(img, conf=conf, imgsz=imgsz)
    if detection is None:
        return None
    class_name, confidence, corners = detection
    corners = expand_quad(ensure_clockwise(corners), pad)
    return class_name, rectify(img, corners), confidence


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def list_images(image_path):
    if os.path.isfile(image_path):
        return [image_path]
    return sorted(p for p in glob.glob(os.path.join(image_path, "*"))
                  if p.lower().endswith(IMAGE_EXTS))


def image_number(path):
    """'img3.jpg' -> '3'. Falls back to the file stem if it has no number."""
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"(\d+)$", stem)
    return match.group(1) if match else stem


def read_settings(config):
    """Task 1 settings from config.txt, falling back to the defaults."""
    config = config or {}
    return (float(config.get("task1_conf", DEFAULT_CONF)),
            int(config.get("task1_imgsz", DEFAULT_IMGSZ)),
            float(config.get("task1_pad", DEFAULT_PAD)))


def run_task1(image_path, config):
    conf, imgsz, pad = read_settings(config)
    images = list_images(image_path)
    print(f"Task 1: {len(images)} image(s) in {image_path} (conf={conf}, imgsz={imgsz}, pad={pad})")

    for path in images:
        img = cv2.imread(path)  # applies EXIF orientation, as Roboflow did
        if img is None:
            print(f"Could not read image: {path}")
            continue

        try:
            found = extract_reading_region(img, conf=conf, imgsz=imgsz, pad=pad)
            if found is None:
                print(f"{os.path.basename(path)}: no device detected, no output")
                continue
            class_name, region, confidence = found
            print(f"{os.path.basename(path)}: {class_name} (conf {confidence:.2f}), "
                  f"region {region.shape[1]}x{region.shape[0]}")
            out_name = f"{PREFIX.get(class_name, class_name)}{image_number(path)}.png"
            save_output(os.path.join(OUTPUT_DIR, out_name), region, output_type='image')
        except Exception as e:
            # One bad image must not stop the rest of the batch.
            print(f"Task 1 failed on {path}: {e}")
