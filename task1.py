

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
Task 1: detect the medical device in each photo and extract its reading region.

Two stages:

1. Learned detection. A YOLOv8n-pose model (trained by train_task1.py, weights
   in data/task1_pose.pt) finds the device, classifies it (bpm or thermo) and
   predicts the four corners of the reading region in the device's own reading
   orientation: 0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left.
   These corners are good enough to locate the region and tell which way is
   up, but are often tens of pixels off

2. Classical refinement of those corners (can be turned off with
   "task1_refine: 0" in config.txt):
     - BPM: the LCD is darker than the white bezel around it, so each of its
       four borders is a long straight dark-to-bright edge. Each border is
       re-found by searching lines near the predicted one (a Hough-style search
       over small shifts and rotations), and the corners are the intersections
       of the four lines
     - Thermometer: the scale area has no physical border, but the glass tube
       runs exactly along the scale. Its direction gives the true rotation; the
       predicted corners give the extent along and across it

The refined corners then drive a perspective warp to an upright rectangle,
which corrects rotation and perspective distortion in one step

  - BPM         -> output/task1/lcdX.png     (the LCD panel)
  - Thermometer -> output/task1/thermoX.png  (the scale area, 50 degC at the top)
  - No device   -> no output file
"""

import glob
import os
import re

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))           # Absolute path of the directory containing this script, so paths resolve from any working directory
MODEL_PATH = os.path.join(HERE, "data", "task1_pose.pt")    # Pose model weights written by train_task1.py
OUTPUT_DIR = os.path.join(HERE, "output", "task1")          # Output folder for the extracted reading regions
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")              # File extensions treated as input images

# Defaults, overridable from config.txt (e.g. "task1_conf: 0.6")
DEFAULT_CONF = 0.5  # Minimum detection confidence; below this the image counts as negative
DEFAULT_IMGSZ = 960 # Inference image size, same as training
DEFAULT_PAD = 0.0   # Grow the corner quad outwards by this fraction of its size
DEFAULT_REFINE = 1  # 1 = refine the predicted corners with classical CV
DEFAULT_DEBUG = 0   # 1 = also save debug_imgX.jpg with the predicted and refined corners

# Edge searches run on a copy of the photo scaled to this longest side:
# fast, and still about 4 px accuracy on a 4000 px photo.
WORK_SIZE = 1000

# Output file prefix for each class name the model predicts.
PREFIX = {"bpm": "lcd", "thermo": "thermo"}

# Cached pose model, loaded on first use by load_model
_model = None


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
# Stage 1: learned detection
# --------------------------------------------------------------------------- #

def load_model():
    """
    load_model: Load the pose model once and reuse it (Task 4 calls this per image)

    :return: the loaded YOLO pose model
    """
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
    detect_device: Find the most confident device in the photo

    Each positive image holds exactly one device, so only the best detection
    is kept; extra boxes would otherwise count as failed detections

    :param img (np.ndarray): BGR photo
    :param conf (float): minimum detection confidence
    :param imgsz (int): inference image size
    :return: (class_name, confidence, corners, box), or None if nothing scores above conf.
             corners is a (4, 2) float array in pixels (TL, TR, BR, BL);
             box is the device's bounding box (x0, y0, x1, y1)
    """
    model = load_model()
    result = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    # Keep only the highest-confidence detection
    best_idx = int(result.boxes.conf.argmax())
    class_name = result.names[int(result.boxes.cls[best_idx])]
    confidence = float(result.boxes.conf[best_idx])
    corners = result.keypoints.xy[best_idx].cpu().numpy().astype(np.float32)
    box = result.boxes.xyxy[best_idx].cpu().numpy().astype(np.float32)

    return class_name, confidence, corners, box


def repair_outlier_corner(corners, box, margin=0.1):
    """
    repair_outlier_corner: Rebuild a single corner that was predicted far outside the
    device box

    The reading region lies inside the device, so a corner well outside the
    detected box is a bad prediction. If exactly one corner is off, it is
    replaced by completing the parallelogram from the other three
    (missing = previous + next - opposite); the refinement step then snaps it
    to the real border. Two or more bad corners cannot be repaired this way

    :param corners (np.ndarray): (4, 2) predicted corners, TL, TR, BR, BL
    :param box (np.ndarray): device bounding box (x0, y0, x1, y1)
    :param margin (float): allowed distance outside the box, as a fraction of its size
    :return: (corners, repaired) where repaired (bool) is True if a corner was rebuilt
    """
    box_left, box_top, box_right, box_bottom = box
    margin_x, margin_y = margin * (box_right - box_left), margin * (box_bottom - box_top)

    # Flag every corner that lies outside the box grown by the margin
    outside = [(corner_x < box_left - margin_x or corner_x > box_right + margin_x
                or corner_y < box_top - margin_y or corner_y > box_bottom + margin_y)
               for corner_x, corner_y in corners]
    
    if sum(outside) != 1:
        return corners, False
    
    bad_idx = outside.index(True)
    repaired = corners.copy()

    # Parallelogram completion: previous + next - opposite
    repaired[bad_idx] = (corners[(bad_idx - 1) % 4] + corners[(bad_idx + 1) % 4]
                         - corners[(bad_idx + 2) % 4])
    
    return repaired, True


def ensure_clockwise(corners):
    """
    ensure_clockwise: Make sure the corners go round clockwise (in image coordinates)

    The model is trained on clockwise TL, TR, BR, BL labels, so this is only a
    safety net: an anticlockwise quad would produce a mirrored output. Keeping
    corner 0 and reversing the direction preserves which side is "top"

    :param corners (np.ndarray): (4, 2) corners
    :return: the corners in clockwise order (np.ndarray)
    """
    corner_xs, corner_ys = corners[:, 0], corners[:, 1]

    # Shoelace formula; a negative area means anticlockwise in image coordinates
    signed_area = 0.5 * np.sum(corner_xs * np.roll(corner_ys, -1) - np.roll(corner_xs, -1) * corner_ys)

    if signed_area < 0:
        corners = corners[[0, 3, 2, 1]]

    return corners


# --------------------------------------------------------------------------- #
# Stage 2: classical corner refinement
# --------------------------------------------------------------------------- #

def gradient_images(img):
    """
    gradient_images: Compute downscaled Sobel x/y gradients of the photo

    :param img (np.ndarray): BGR photo
    :return: (grad_x, grad_y, scale), the gradient images (np.ndarray) and the
             downscale factor used (float)
    """
    # Scale so the longest side is WORK_SIZE pixels
    scale = WORK_SIZE / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Light blur so the gradients follow real edges rather than pixel noise
    gray = cv2.GaussianBlur(gray, (0, 0), 1.5)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    return grad_x, grad_y, scale


def rotate(vector, radians):
    """
    rotate: Rotate a 2D vector by the given angle

    :param vector (np.ndarray): 2D vector (x, y)
    :param radians (float): rotation angle in radians
    :return: the rotated vector (np.ndarray, float32)
    """
    cos_a, sin_a = np.cos(radians), np.sin(radians)

    return np.array([cos_a * vector[0] - sin_a * vector[1], sin_a * vector[0] + cos_a * vector[1]],
                    dtype=np.float32)


def search_line(grad_x, grad_y, centre, direction, half_length, max_shift, max_angle_deg,
                angle_step_deg, score_fn, num_samples=150):
    """
    search_line: Find the straight line near (centre, direction) with the best edge score

    A restricted Hough transform: candidate lines are the input line rotated by
    up to +-max_angle_deg and shifted sideways by up to +-max_shift pixels. For
    each candidate the image gradient is sampled at num_samples points along it,
    projected on the line's normal, and reduced to one score by score_fn

    :param grad_x (np.ndarray): x gradient image
    :param grad_y (np.ndarray): y gradient image
    :param centre (np.ndarray): point on the starting line
    :param direction (np.ndarray): unit direction of the starting line
    :param half_length (float): half the length of line sampled, in pixels
    :param max_shift (float): largest sideways shift tried, in pixels
    :param max_angle_deg (float): largest rotation tried, in degrees
    :param angle_step_deg (float): rotation step, in degrees
    :param score_fn (callable): maps the (offsets, samples) gradient array to one score per offset
    :param num_samples (int): number of sample points along each line
    :return: (point_on_line, unit_direction) of the best line
    """
    # Positions along the line and sideways offsets to try
    positions = np.linspace(-1.0, 1.0, num_samples) * half_length
    offsets = np.arange(-max_shift, max_shift + 0.5, 0.5)
    best_score, best_line = -np.inf, (centre, direction)

    for angle in np.radians(np.arange(-max_angle_deg, max_angle_deg + 1e-6, angle_step_deg)):
        rotated_dir = rotate(direction, angle)
        normal = np.array([-rotated_dir[1], rotated_dir[0]], dtype=np.float32)

        # Sample grid: one row per sideways offset, one column per point along the line.
        sample_points = (centre[None, None, :] + offsets[:, None, None] * normal[None, None, :]
                         + positions[None, :, None] * rotated_dir[None, None, :])
        
        map_x = sample_points[..., 0].astype(np.float32)
        map_y = sample_points[..., 1].astype(np.float32)
        grad_x_samples = cv2.remap(grad_x, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
        grad_y_samples = cv2.remap(grad_y, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)

        # Gradient across the line
        across = grad_x_samples * normal[0] + grad_y_samples * normal[1]
        scores = score_fn(across)
        best_offset_idx = int(np.argmax(scores))

        if scores[best_offset_idx] > best_score:
            best_score = scores[best_offset_idx]
            best_line = (centre + offsets[best_offset_idx] * normal, rotated_dir)
             
    return best_line


def intersect_lines(point1, dir1, point2, dir2):
    """
    intersect_lines: Intersection of the lines point1 + t*dir1 and point2 + u*dir2

    :param point1 (np.ndarray): point on the first line
    :param dir1 (np.ndarray): direction of the first line
    :param point2 (np.ndarray): point on the second line
    :param dir2 (np.ndarray): direction of the second line
    :return: the intersection point (np.ndarray)
    """
    # Solve point1 + t*dir1 = point2 + u*dir2 for (t, u)
    line_params = np.linalg.solve(np.array([dir1, -dir2]).T, point2 - point1)
    return point1 + line_params[0] * dir1


def is_plausible(refined, predicted, max_move=0.25, area_range=(0.6, 1.4)):
    """
    is_plausible: Reject a refinement that wanders far from the model's quad

    :param refined (np.ndarray): (4, 2) refined corners
    :param predicted (np.ndarray): (4, 2) predicted corners
    :param max_move (float): largest corner move allowed, as a fraction of the quad diagonal
    :param area_range (tuple): (min, max) allowed ratio of refined area to predicted area
    :return: True if the refined quad is convex, close to the prediction and of similar area (bool)
    """
    if not cv2.isContourConvex(refined.reshape(-1, 1, 2).astype(np.float32)):
        return False
    
    diagonal = np.linalg.norm(predicted[2] - predicted[0])

    if np.max(np.linalg.norm(refined - predicted, axis=1)) > max_move * diagonal:
        return False
    area_ratio = cv2.contourArea(refined) / max(cv2.contourArea(predicted), 1e-6)

    return area_range[0] <= area_ratio <= area_range[1]


def refine_lcd_corners(img, corners):
    """
    refine_lcd_corners: Snap each side of the predicted LCD quad to the real LCD border

    The LCD is darker than the bezel, so its border is where brightness rises
    going outwards. Only that sign of edge is scored, which ignores the bezel's
    outer edge and most digit edges; averaging along the whole side favours the
    long border over short digit segments

    :param img (np.ndarray): BGR photo
    :param corners (np.ndarray): (4, 2) predicted corners, TL, TR, BR, BL
    :return: (4, 2) refined corners in photo pixels (np.ndarray, float32)
    """
    grad_x, grad_y, scale = gradient_images(img)
    # Work in the downscaled gradient image
    quad = corners * scale
    centre = quad.mean(axis=0)
    size = np.linalg.norm(quad[2] - quad[0])

    def outward_rise(across):
        """
        outward_rise: Score each offset by the mean brightness rise going outwards

        :param across (np.ndarray): gradient across the line, one row per offset
        :return: one score per offset (np.ndarray)
        """
        return np.clip(across, 0, None).mean(axis=1)

    lines = []
    for side_idx in range(4):
        start, end = quad[side_idx], quad[(side_idx + 1) % 4]
        midpoint = (start + end) / 2
        direction = (end - start) / np.linalg.norm(end - start)

        # search_line's normal is (-dy, dx); flip the side so it points outwards
        if np.dot(np.array([-direction[1], direction[0]]), midpoint - centre) < 0:
            direction = -direction

        lines.append(search_line(
            grad_x,
            grad_y,
            midpoint,
            direction,
            0.4 * np.linalg.norm(end - start),  # middle 80% of the side: LCD corners are rounded
            max_shift=0.25 * size,
            max_angle_deg=35,                   # strong perspective can leave the quad ~30 deg off (validation img4)
            angle_step_deg=1.0,
            score_fn=outward_rise,
        ))

    # Each corner is where the side before it meets the side after it
    refined = np.array([intersect_lines(*lines[side_idx - 1], *lines[side_idx]) for side_idx in range(4)],
                       dtype=np.float32) / scale
    
    return refined


def refine_thermo_corners(img, corners):
    """
    refine_thermo_corners: Re-align the predicted scale quad with the thermometer's
    glass tube

    The tube's edges are the longest straight lines along the scale, so the
    strongest line near the quad's centre line gives the true direction. The
    output rectangle keeps the predicted extent along and across that direction
    (taking the outermost corner on each side, so no markings are lost) and the
    predicted "top" end (corners 0/1, the 50 degC end)

    :param img (np.ndarray): BGR photo
    :param corners (np.ndarray): (4, 2) predicted corners, TL, TR, BR, BL
    :return: (4, 2) re-aligned corners in photo pixels (np.ndarray, float32)
    """
    grad_x, grad_y, scale = gradient_images(img)

    # Work in the downscaled gradient image
    quad = corners * scale
    centre = quad.mean(axis=0)
    top, bottom = (quad[0] + quad[1]) / 2, (quad[2] + quad[3]) / 2
    length = np.linalg.norm(bottom - top)
    width = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    axis = (bottom - top) / length

    def edge_strength(across):
        """
        edge_strength: Score each offset by the mean gradient magnitude across the line

        :param across (np.ndarray): gradient across the line, one row per offset
        :return: one score per offset (np.ndarray)
        """
        # Either edge of the tube counts, so the sign is ignored
        return np.abs(across).mean(axis=1)

    _, tube_dir = search_line(
        grad_x,
        grad_y,
        centre,
        axis,
        0.4 * length,               # middle 80% of the scale length
        max_shift=0.35 * width,
        max_angle_deg=15,
        angle_step_deg=0.5,
        score_fn=edge_strength,
    )

    # Keep the direction pointing from top to bottom
    if np.dot(tube_dir, axis) < 0:
        tube_dir = -tube_dir
    normal = np.array([-tube_dir[1], tube_dir[0]], dtype=np.float32)

    # Extents of the predicted corners in the new (along, across) frame.
    corner_centre = corners.mean(axis=0)
    along = (corners - corner_centre) @ tube_dir
    across = (corners - corner_centre) @ normal
    along_top, along_bottom = along.min(), along.max()

    # Corner 0 (top-left) decides which side of the axis is "left".
    left_sign = np.sign((corners[0] - (corners[0] + corners[1]) / 2) @ normal) or 1.0
    across_left = across.max() if left_sign > 0 else across.min()
    across_right = across.min() if left_sign > 0 else across.max()

    def point(along_pos, across_pos):
        """
        point: Convert (along, across) coordinates back to photo pixels

        :param along_pos (float): distance along the tube from the corner centre
        :param across_pos (float): distance across the tube from the corner centre
        :return: the point in photo pixels (np.ndarray)
        """
        return corner_centre + along_pos * tube_dir + across_pos * normal

    return np.array([point(along_top, across_left), point(along_top, across_right),
                     point(along_bottom, across_right), point(along_bottom, across_left)],
                    dtype=np.float32)


def refine_corners(img, class_name, corners):
    """
    refine_corners: Refine the corners for this class; keep the prediction if the result
    looks wrong

    :param img (np.ndarray): BGR photo
    :param class_name (str): predicted class, "bpm" or "thermo"
    :param corners (np.ndarray): (4, 2) predicted corners, TL, TR, BR, BL
    :return: (corners, refined) where refined (bool) is True if the refined corners were kept
    """
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
        # Parallel lines or an OpenCV failure: fall back to the prediction
        return corners, False
    
    refined = ensure_clockwise(refined)

    if not is_plausible(refined, corners, **limits):
        return corners, False
    
    return refined, True


# --------------------------------------------------------------------------- #
# Geometric correction
# --------------------------------------------------------------------------- #

def expand_quad(corners, pad_frac):
    """
    expand_quad: Push each corner away from the centre by pad_frac (0 = unchanged)

    :param corners (np.ndarray): (4, 2) corners
    :param pad_frac (float): fraction of the corner-to-centre distance to add
    :return: the expanded corners (np.ndarray)
    """
    if pad_frac <= 0:
        return corners
    centre = corners.mean(axis=0)

    return centre + (corners - centre) * (1.0 + pad_frac)


def rectify(img, corners):
    """
    rectify: Warp the quad TL, TR, BR, BL to an upright rectangle

    The output size is the longer of each pair of opposite edges, so the
    region keeps (roughly) its original resolution and aspect ratio

    :param img (np.ndarray): BGR photo
    :param corners (np.ndarray): (4, 2) corners, TL, TR, BR, BL
    :return: the rectified region (np.ndarray)
    """
    top_left, top_right, bottom_right, bottom_left = corners
    width = int(round(max(np.linalg.norm(top_right - top_left),
                          np.linalg.norm(bottom_right - bottom_left))))
    height = int(round(max(np.linalg.norm(bottom_left - top_left),
                           np.linalg.norm(bottom_right - top_right))))
    
    if width < 2 or height < 2:
        raise ValueError(f"degenerate reading region ({width}x{height})")

    # Target corners of the upright rectangle, in the same TL, TR, BR, BL order
    target_corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                              dtype=np.float32)
    homography = cv2.getPerspectiveTransform(corners.astype(np.float32), target_corners)

    return cv2.warpPerspective(
        img,
        homography,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,    # corners slightly outside the photo leave no black wedges
    )


def extract_reading_region(img, conf=DEFAULT_CONF, imgsz=DEFAULT_IMGSZ, pad=DEFAULT_PAD,
                           refine=True, debug_path=None):
    """
    extract_reading_region: Run the full Task 1 step for one image (shared with Task 4)

    :param img (np.ndarray): BGR photo
    :param conf (float): minimum detection confidence
    :param imgsz (int): inference image size
    :param pad (float): fraction to grow the corner quad by before warping
    :param refine (bool): refine the predicted corners with classical CV
    :param debug_path (str): if given, save a debug image of the corners here
    :return: (class_name, region_image, confidence), or None for a negative image
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
    """
    save_debug: Save the photo with the model's corners (red) and the refined corners (green)

    :param img (np.ndarray): BGR photo
    :param predicted (np.ndarray): (4, 2) predicted corners
    :param refined (np.ndarray): (4, 2) refined corners, or None if refinement was not used
    :param path (str): destination image path
    :return: None
    """
    annotated = img.copy()

    # Line thickness scaled to the photo width
    thickness = max(2, img.shape[1] // 500)
    cv2.polylines(annotated, [predicted.astype(np.int32)], True, (0, 0, 255), thickness)
    if refined is not None:
        cv2.polylines(annotated, [refined.astype(np.int32)], True, (0, 255, 0), thickness)

    # Number each final corner 0-3
    for corner_idx, (corner_x, corner_y) in enumerate(refined if refined is not None else predicted):
        cv2.putText(
            annotated,
            str(corner_idx),
            (int(corner_x), int(corner_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            img.shape[1] / 1000,    # font scale relative to the photo width
            (255, 0, 255),          # magenta
            thickness,
        )

    save_output(path, annotated, output_type='image')


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def list_images(image_path):
    """
    list_images: List the input images at image_path

    :param image_path (str): a single image file or a folder of images
    :return: sorted list of image paths (list of str)
    """
    if os.path.isfile(image_path):
        return [image_path]
    return sorted(path for path in glob.glob(os.path.join(image_path, "*"))
                  if path.lower().endswith(IMAGE_EXTS))


def image_number(path):
    """
    image_number: Extract the trailing number of an image name ('img3.jpg' -> '3'),
    falling back to the file stem if it has no number

    :param path (str): image path
    :return: the number, or the file stem (str)
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"(\d+)$", stem)
    return match.group(1) if match else stem


def read_settings(config):
    """
    read_settings: Read the Task 1 settings from config.txt, falling back to the defaults

    :param config (dict): key/value pairs from config.txt, or None
    :return: dict with "conf", "imgsz", "pad", "refine" and "debug"
    """
    config = config or {}
    return {
        "conf": float(config.get("task1_conf", DEFAULT_CONF)),
        "imgsz": int(config.get("task1_imgsz", DEFAULT_IMGSZ)),
        "pad": float(config.get("task1_pad", DEFAULT_PAD)),
        "refine": int(config.get("task1_refine", DEFAULT_REFINE)) != 0,
        "debug": int(config.get("task1_debug", DEFAULT_DEBUG)) != 0,
    }


def run_task1(image_path, config):
    """
    run_task1: Extract the reading region of every image and save it to output/task1

    :param image_path (str): a single image file or a folder of images
    :param config (dict): key/value pairs from config.txt
    :return: None
    """
    print("=============================================================")
    print("TASK 1 - Detecting devices and extracting reading regions")
    print("=============================================================")

    settings = read_settings(config)
    images = list_images(image_path)
    print(f"Note: {len(images)} image(s) in {image_path} ({settings})")

    for path in images:
        # cv2.imread applies the EXIF orientation, as Roboflow did
        img = cv2.imread(path)
        if img is None:
            print(f"Could not read image: {path}")
            continue

        name = os.path.basename(path)
        number = image_number(path)
        debug_path = (os.path.join(OUTPUT_DIR, "debug", f"debug_img{number}.jpg")
                      if settings["debug"] else None)
        try:
            print(f"[{name}]:")
            extraction = extract_reading_region(
                img,
                conf=settings["conf"],
                imgsz=settings["imgsz"],
                pad=settings["pad"],
                refine=settings["refine"],
                debug_path=debug_path,      # None when debugging is off
            )
            if extraction is None:
                print(f"  no device detected, no output")
                continue
            class_name, region, confidence = extraction
            print(f"  {class_name} (conf {confidence:.2f}), region {region.shape[1]}x{region.shape[0]}")
            # e.g. lcd2.png or thermo1.png
            out_name = f"{PREFIX.get(class_name, class_name)}{number}.png"
            save_output(os.path.join(OUTPUT_DIR, out_name), region, output_type='image')
            print("------")
        except Exception as error:
            # One bad image must not stop the rest of the batch.
            print(f"Task 1 failed on {path}: {error}")
