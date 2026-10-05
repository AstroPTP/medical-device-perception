

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
Task 1: detect the medical device in each photo and extract its reading region.

Two stages:

1. Learned detection. A YOLOv8n-pose model (trained by train_task1.py, weights
   in data/task1_pose.pt) finds the device, classifies it (bpm or thermo) and
   predicts the four corners of the reading region in the device's own reading
   orientation: 0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left.
   These corners are good enough to locate the region and tell which way is
   up, but are often tens of pixels off.

2. Classical refinement of those corners (can be turned off with
   "task1_refine: 0" in config.txt):
     - BPM: the LCD is darker than the white bezel around it, so each of its
       four borders is a long straight dark-to-bright edge. Each border is
       re-found by searching lines near the predicted one (a Hough-style search
       over small shifts and rotations), and the corners are the intersections
       of the four lines.
     - Thermometer: the scale area has no physical border, but the glass tube
       runs exactly along the scale. Its direction gives the true rotation; the
       predicted corners give the extent along and across it.

The refined corners then drive a perspective warp to an upright rectangle,
which corrects rotation and perspective distortion in one step.

  - BPM         -> output/task1/lcdX.png     (the LCD panel)
  - Thermometer -> output/task1/thermoX.png  (the scale area, 50 degC at the top)
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
DEFAULT_REFINE = 1     # 1 = refine the predicted corners with classical CV
DEFAULT_DEBUG = 0      # 1 = also save debug_imgX.jpg with predicted/refined corners

# Edge searches run on a copy of the photo scaled to this longest side:
# fast, and still about 4 px accuracy on a 4000 px photo.
WORK_SIZE = 1000

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
# Stage 1: learned detection
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
    Return (class_name, confidence, corners, box) for the most confident
    device, or None if nothing scores above `conf`.

    Each positive image holds exactly one device, so only the best detection
    is kept; extra boxes would otherwise count as failed detections.
    corners is a (4, 2) float array in pixel coordinates: TL, TR, BR, BL.
    box is the device's bounding box (x0, y0, x1, y1).
    """
    model = load_model()
    result = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best = int(result.boxes.conf.argmax())
    class_name = result.names[int(result.boxes.cls[best])]
    confidence = float(result.boxes.conf[best])
    corners = result.keypoints.xy[best].cpu().numpy().astype(np.float32)
    box = result.boxes.xyxy[best].cpu().numpy().astype(np.float32)
    return class_name, confidence, corners, box


def repair_outlier_corner(corners, box, margin=0.1):
    """
    Rebuild a single corner that was predicted far outside the device box.

    The reading region lies inside the device, so a corner well outside the
    detected box is a bad prediction. If exactly one corner is off, it is
    replaced by completing the parallelogram from the other three
    (missing = previous + next - opposite); the refinement step then snaps it
    to the real border. Two or more bad corners cannot be repaired this way.
    """
    x0, y0, x1, y1 = box
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    outside = [(x < x0 - mx or x > x1 + mx or y < y0 - my or y > y1 + my) for x, y in corners]
    if sum(outside) != 1:
        return corners, False
    i = outside.index(True)
    repaired = corners.copy()
    repaired[i] = corners[(i - 1) % 4] + corners[(i + 1) % 4] - corners[(i + 2) % 4]
    return repaired, True


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


# --------------------------------------------------------------------------- #
# Stage 2: classical corner refinement
# --------------------------------------------------------------------------- #

def gradient_images(img):
    """Downscaled Sobel x/y gradients of the photo, and the scale factor used."""
    s = WORK_SIZE / max(img.shape[:2])
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.5)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy, s


def rotate(v, radians):
    c, s = np.cos(radians), np.sin(radians)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float32)


def search_line(gx, gy, centre, direction, half_len, max_shift, max_angle_deg,
                angle_step_deg, score_fn, n_samples=150):
    """
    Find the straight line near (centre, direction) with the best edge score.

    A restricted Hough transform: candidate lines are the input line rotated by
    up to +-max_angle_deg and shifted sideways by up to +-max_shift pixels. For
    each candidate the image gradient is sampled at n_samples points along it,
    projected on the line's normal, and reduced to one score by score_fn.

    Returns (point_on_line, unit_direction).
    """
    t = np.linspace(-1.0, 1.0, n_samples) * half_len
    offsets = np.arange(-max_shift, max_shift + 0.5, 0.5)
    best_score, best_line = -np.inf, (centre, direction)

    for angle in np.radians(np.arange(-max_angle_deg, max_angle_deg + 1e-6, angle_step_deg)):
        d = rotate(direction, angle)
        normal = np.array([-d[1], d[0]], dtype=np.float32)
        # Sample grid: one row per sideways offset, one column per point along the line.
        pts = (centre[None, None, :] + offsets[:, None, None] * normal[None, None, :]
               + t[None, :, None] * d[None, None, :])
        map_x, map_y = pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32)
        vx = cv2.remap(gx, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
        vy = cv2.remap(gy, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
        across = vx * normal[0] + vy * normal[1]   # gradient across the line
        scores = score_fn(across)
        k = int(np.argmax(scores))
        if scores[k] > best_score:
            best_score = scores[k]
            best_line = (centre + offsets[k] * normal, d)
    return best_line


def intersect_lines(p1, d1, p2, d2):
    """Intersection of the lines p1 + t*d1 and p2 + u*d2."""
    t = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
    return p1 + t[0] * d1


def is_plausible(refined, predicted, max_move=0.25, area_range=(0.6, 1.4)):
    """Reject a refinement that wanders far from the model's quad."""
    if not cv2.isContourConvex(refined.reshape(-1, 1, 2).astype(np.float32)):
        return False
    diag = np.linalg.norm(predicted[2] - predicted[0])
    if np.max(np.linalg.norm(refined - predicted, axis=1)) > max_move * diag:
        return False
    ratio = cv2.contourArea(refined) / max(cv2.contourArea(predicted), 1e-6)
    return area_range[0] <= ratio <= area_range[1]


def refine_lcd_corners(img, corners):
    """
    Snap each side of the predicted LCD quad to the real LCD border.

    The LCD is darker than the bezel, so its border is where brightness rises
    going outwards. Only that sign of edge is scored, which ignores the bezel's
    outer edge and most digit edges; averaging along the whole side favours the
    long border over short digit segments.
    """
    gx, gy, s = gradient_images(img)
    quad = corners * s
    centre = quad.mean(axis=0)
    size = np.linalg.norm(quad[2] - quad[0])

    def outward_rise(across):
        return np.clip(across, 0, None).mean(axis=1)

    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        mid = (a + b) / 2
        d = (b - a) / np.linalg.norm(b - a)
        # search_line's normal is (-dy, dx); flip the side so it points outwards.
        if np.dot(np.array([-d[1], d[0]]), mid - centre) < 0:
            d = -d
        # Sample the middle 80% of the side: LCD corners are rounded.
        # +-35 deg: on strong perspective views the predicted quad can be
        # rotated ~30 deg less than the real LCD (validation img4).
        lines.append(search_line(gx, gy, mid, d, 0.4 * np.linalg.norm(b - a),
                                 max_shift=0.25 * size, max_angle_deg=35, angle_step_deg=1.0,
                                 score_fn=outward_rise))

    refined = np.array([intersect_lines(*lines[i - 1], *lines[i]) for i in range(4)],
                       dtype=np.float32) / s
    return refined


def refine_thermo_corners(img, corners):
    """
    Re-align the predicted scale quad with the thermometer's glass tube.

    The tube's edges are the longest straight lines along the scale, so the
    strongest line near the quad's centre line gives the true direction. The
    output rectangle keeps the predicted extent along and across that direction
    (taking the outermost corner on each side, so no markings are lost) and the
    predicted "top" end (corners 0/1, the 50 degC end).
    """
    gx, gy, s = gradient_images(img)
    quad = corners * s
    centre = quad.mean(axis=0)
    top, bottom = (quad[0] + quad[1]) / 2, (quad[2] + quad[3]) / 2
    length = np.linalg.norm(bottom - top)
    width = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    axis = (bottom - top) / length

    def edge_strength(across):
        return np.abs(across).mean(axis=1)   # either edge of the tube

    _, d = search_line(gx, gy, centre, axis, 0.4 * length, max_shift=0.35 * width,
                       max_angle_deg=15, angle_step_deg=0.5, score_fn=edge_strength)
    if np.dot(d, axis) < 0:
        d = -d                                # keep pointing from top to bottom
    normal = np.array([-d[1], d[0]], dtype=np.float32)

    # Extents of the predicted corners in the new (along, across) frame.
    c = corners.mean(axis=0)
    along = (corners - c) @ d
    across = (corners - c) @ normal
    a_top, a_bottom = along.min(), along.max()
    # Corner 0 (top-left) decides which side of the axis is "left".
    left_sign = np.sign((corners[0] - (corners[0] + corners[1]) / 2) @ normal) or 1.0
    w_left = across.max() if left_sign > 0 else across.min()
    w_right = across.min() if left_sign > 0 else across.max()

    def point(a, w):
        return c + a * d + w * normal

    return np.array([point(a_top, w_left), point(a_top, w_right),
                     point(a_bottom, w_right), point(a_bottom, w_left)], dtype=np.float32)


def refine_corners(img, class_name, corners):
    """Refine the corners for this class; keep the prediction if the result looks wrong."""
    try:
        if class_name == "bpm":
            refined = refine_lcd_corners(img, corners)
            limits = dict(max_move=0.25, area_range=(0.6, 1.4))
        elif class_name == "thermo":
            refined = refine_thermo_corners(img, corners)
            # The re-aligned rectangle encloses the predicted quad, so it is
            # always somewhat larger; only reject a gross blow-up.
            limits = dict(max_move=0.3, area_range=(0.8, 2.0))
        else:
            return corners, False
    except (np.linalg.LinAlgError, cv2.error):
        return corners, False
    refined = ensure_clockwise(refined)
    if not is_plausible(refined, corners, **limits):
        return corners, False
    return refined, True


# --------------------------------------------------------------------------- #
# Geometric correction
# --------------------------------------------------------------------------- #

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


def extract_reading_region(img, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ, pad=DEFAULT_PAD,
                           refine=True, debug_path=None):
    """
    Full Task 1 step for one image.

    Returns (class_name, region_image, confidence), or None for a negative image.
    Shared with Task 4.
    """
    detection = detect_device(img, conf=conf, imgsz=imgsz)
    if detection is None:
        return None
    class_name, confidence, predicted, box = detection
    predicted, repaired = repair_outlier_corner(predicted, box)
    if repaired:
        print("  one corner was far outside the device box; rebuilt from the other three")
    predicted = ensure_clockwise(predicted)

    corners, refined = (refine_corners(img, class_name, predicted) if refine
                        else (predicted, False))
    print(f"  corners: {'refined' if refined else 'model prediction'}")
    if debug_path:
        save_debug(img, predicted, corners if refined else None, debug_path)

    corners = expand_quad(corners, pad)
    return class_name, rectify(img, corners), confidence


def save_debug(img, predicted, refined, path):
    """Photo with the model's corners (red) and the refined corners (green)."""
    vis = img.copy()
    thick = max(2, img.shape[1] // 500)
    cv2.polylines(vis, [predicted.astype(np.int32)], True, (0, 0, 255), thick)
    if refined is not None:
        cv2.polylines(vis, [refined.astype(np.int32)], True, (0, 255, 0), thick)
    for i, (x, y) in enumerate(refined if refined is not None else predicted):
        cv2.putText(vis, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    img.shape[1] / 1000, (255, 0, 255), thick)
    save_output(path, vis, output_type='image')


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
    return {
        "conf": float(config.get("task1_conf", DEFAULT_CONF)),
        "imgsz": int(config.get("task1_imgsz", DEFAULT_IMGSZ)),
        "pad": float(config.get("task1_pad", DEFAULT_PAD)),
        "refine": int(config.get("task1_refine", DEFAULT_REFINE)) != 0,
        "debug": int(config.get("task1_debug", DEFAULT_DEBUG)) != 0,
    }


def run_task1(image_path, config):
    settings = read_settings(config)
    images = list_images(image_path)
    print(f"Task 1: {len(images)} image(s) in {image_path} ({settings})")

    for path in images:
        img = cv2.imread(path)  # applies EXIF orientation, as Roboflow did
        if img is None:
            print(f"Could not read image: {path}")
            continue

        name = os.path.basename(path)
        number = image_number(path)
        debug_path = (os.path.join(OUTPUT_DIR, "debug", f"debug_img{number}.jpg")
                      if settings["debug"] else None)
        try:
            print(f"{name}:")
            found = extract_reading_region(img, conf=settings["conf"], imgsz=settings["imgsz"],
                                           pad=settings["pad"], refine=settings["refine"],
                                           debug_path=debug_path)
            if found is None:
                print(f"  no device detected, no output")
                continue
            class_name, region, confidence = found
            print(f"  {class_name} (conf {confidence:.2f}), region {region.shape[1]}x{region.shape[0]}")
            out_name = f"{PREFIX.get(class_name, class_name)}{number}.png"
            save_output(os.path.join(OUTPUT_DIR, out_name), region, output_type='image')
        except Exception as e:
            # One bad image must not stop the rest of the batch.
            print(f"Task 1 failed on {path}: {e}")
