

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
Task 2: segment the reading inside a rectified device image (output of Task 1)

  - BPM LCD     -> one crop per seven-segment digit, saved as d1.png, d2.png, ...
                   in reading order (systolic row, diastolic row, pulse row,
                   each left to right)
  - Thermometer -> one crop, t.png, centred on the top of the liquid column and
                   reaching past the numbered scale labels above and below it

Everything here is classical OpenCV: the input is already upright and tightly
cropped, so no learned model is needed. All sizes are relative to the image
size so the code does not depend on the input resolution

Set "task2_debug: 1" in config.txt to also save output/task2/debug/<name>.png,
showing the detected digit boxes, or the liquid top, scale labels and ticks
"""

import glob
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # Absolute path of the directory containing this script, so paths resolve from any working directory
OUTPUT_DIR = os.path.join(HERE, "output", "task2")  # Output folder for the digit and scale crops
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")      # File extensions treated as input images

DEFAULT_DEBUG = 0 # 1 = also save a debug image per input

# An upright thermometer plate is much taller than it is wide; a BPM LCD is not.
# Only used when the file name does not say which device it is.
THERMO_MIN_ASPECT = 2.5

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


def dark_print_mask(img):
    """
    dark_print_mask: Build a binary mask of dark print or LCD segments, with uneven
    lighting removed

    :param img (np.ndarray): BGR image
    :return: mask (np.ndarray, uint8) where 255 = dark print or segment
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    width = gray.shape[1]

    # Dividing by a heavily blurred copy flattens shadows and glare gradients.
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=width / 10)
    flat = cv2.divide(gray, background, scale=255)

    # Gaussian blur + Otsu's thresholding, then invert so the dark print is
    # the white foreground for connected component analysis (as in prac04).
    blurred = cv2.GaussianBlur(flat, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.bitwise_not(binary)


# --------------------------------------------------------------------------- #
# LCD digits
# --------------------------------------------------------------------------- #

def lcd_segment_mask(img):
    """
    lcd_segment_mask: Build a mask of the dark LCD segments, with single-pixel noise removed

    :param img (np.ndarray): BGR LCD image
    :return: mask (np.ndarray, uint8) where 255 = segment
    """
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Morphological opening removes isolated noise pixels
    return cv2.morphologyEx(dark_print_mask(img), cv2.MORPH_OPEN, element)


def merge_fragments(boxes, max_height, shape):
    """
    merge_fragments: Join boxes that are pieces of one digit

    At low resolution the gap between a digit's upper and lower segments can
    be wider than the dilation bridges, so a "1" or "7" falls apart into a top
    and a bottom piece. Pieces of one digit share their columns, and together
    are no taller than a whole digit; a digit and the one in the row below it
    also share columns, but together are about twice as tall. Only sizeable
    pieces separated by a thin gap are joined, so specks next to an icon or
    strips of bezel along the border are not stitched into a digit

    :param boxes (list): boxes as (x, y, w, h)
    :param max_height (float): height of the tallest whole digit
    :param shape (tuple): image shape, used for the border test
    :return: list of boxes (x, y, w, h) after merging
    """
    def is_piece(box):
        """
        is_piece: Check whether a box is large enough to be part of a digit and clear
        of the border

        :param box (list): box as (x, y, w, h)
        :return: True if the box can be merged (bool)
        """
        return box[3] >= 0.2 * max_height and not touches_border(box, shape)

    boxes = [list(box) for box in boxes]
    merged = True
    # Repeat until no pair can be merged; restart after every merge
    while merged:
        merged = False
    
        for idx_a in range(len(boxes)):
            for idx_b in range(idx_a + 1, len(boxes)):
                left_a, top_a, width_a, height_a = boxes[idx_a]
                left_b, top_b, width_b, height_b = boxes[idx_b]

                # Only a pair of digit pieces is tested for merging
                if is_piece(boxes[idx_a]) and is_piece(boxes[idx_b]):
                    # Shared columns, and the vertical gap between the two boxes
                    overlap = min(left_a + width_a, left_b + width_b) - max(left_a, left_b)
                    gap = max(top_a, top_b) - min(top_a + height_a, top_b + height_b)

                    # Union of the two boxes
                    union_left, union_top = min(left_a, left_b), min(top_a, top_b)
                    union_right = max(left_a + width_a, left_b + width_b)
                    union_bottom = max(top_a + height_a, top_b + height_b)

                    if (overlap >= 0.5 * min(width_a, width_b) and gap <= 0.15 * max_height
                            and union_bottom - union_top <= 1.1 * max_height):
                        boxes[idx_a] = [union_left, union_top, union_right - union_left, union_bottom - union_top]
                        del boxes[idx_b]
                        merged = True
                        break

            if merged:
                break

    return [tuple(box) for box in boxes]


def group_rows(boxes):
    """
    group_rows: Group boxes into rows by vertical centre

    :param boxes (list): boxes as (x, y, w, h)
    :return: list of rows top to bottom, each a list of boxes left to right
    """
    boxes = sorted(boxes, key=lambda box: box[1] + box[3] / 2)
    rows = []

    for box in boxes:
        centre_y = box[1] + box[3] / 2
        # Same row if the centre is within half a box height of the row's centre
        if rows and abs(centre_y - rows[-1][0]) < 0.5 * box[3]:
            rows[-1][1].append(box)
        else:
            rows.append([centre_y, [box]])

    return [sorted(row, key=lambda box: box[0]) for _, row in rows]


def touches_border(box, shape, margin=2):
    """
    touches_border: Check whether a box lies within margin pixels of the image border

    :param box (tuple): box as (x, y, w, h)
    :param shape (tuple): image shape
    :param margin (int): distance from the border, in pixels
    :return: True if the box touches the border (bool)
    """
    box_x, box_y, box_width, box_height = box
    img_height, img_width = shape[:2]

    return (box_x <= margin or box_y <= margin or box_x + box_width >= img_width - margin
            or box_y + box_height >= img_height - margin)


def clean_row(row, shape):
    """
    clean_row: Drop boxes that do not fit the row

    The digits of one row are all the same height. A sliver of bezel or shadow
    along the image border can look digit-shaped on its own, but it is not the
    height of the digits next to it. Border boxes are only kept when they match
    the digits away from the border, so a digit clipped by Task 1 survives but a
    lone border sliver does not

    :param row (list): boxes (x, y, w, h) of one row
    :param shape (tuple): image shape
    :return: the boxes of the row that match the typical digit height (list)
    """
    inner = [box for box in row if not touches_border(box, shape)]
    if not inner:
        return []
    
    # Typical digit height, taken from boxes away from the border
    ref_height = float(np.median([box[3] for box in inner]))

    return [box for box in row if abs(box[3] - ref_height) <= 0.25 * ref_height]


def find_lcd_digits(img):
    """
    find_lcd_digits: Find the seven-segment digit boxes on a rectified LCD image

    :param img (np.ndarray): BGR LCD image
    :return: list of rows in reading order, each a list of boxes (x, y, w, h)
    """
    mask = lcd_segment_mask(img)
    height, width = mask.shape

    # Segments of one digit are separated by thin gaps; digits by much wider
    # ones. A small dilation joins each digit's segments into one blob
    kernel_size = max(3, int(round(width * 0.025)))
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    joined = cv2.dilate(mask, element)
    num_labels, label_image, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)

    blobs = []
    # Label 0 is the background
    for label in range(1, num_labels):
        # The LCD holds three rows of digits, so a real digit is well under
        # half the image height. Only such blobs are kept; bezel edges and
        # shadows along the border are far taller (or wider, caught by the
        # aspect test below)
        if stats[label, cv2.CC_STAT_HEIGHT] <= 0.45 * height:
            # Tight box from the undilated segment pixels of this blob, using its
            # x and y distributions (column and row pixel counts)
            blob_x, blob_y, blob_width, blob_height, _ = stats[label]
            blob = ((label_image[blob_y:blob_y + blob_height, blob_x:blob_x + blob_width] == label)
                    & (mask[blob_y:blob_y + blob_height, blob_x:blob_x + blob_width] > 0))

            x_distribution = np.sum(blob, axis=0)
            y_distribution = np.sum(blob, axis=1)
            ink_cols, ink_rows = np.nonzero(x_distribution)[0], np.nonzero(y_distribution)[0]

            # Only a blob with segment pixels gives a box
            if len(ink_cols) > 0:
                blobs.append((blob_x + ink_cols[0], blob_y + ink_rows[0],
                              ink_cols[-1] + 1 - ink_cols[0], ink_rows[-1] + 1 - ink_rows[0]))

    # Seven-segment digits are clearly taller than wide; icons like "OK" or
    # the heart are roughly square. Even a "1" is a solid bar, though, unlike
    # the hairline slivers of bezel left along the border
    def digit_shaped(box):
        """
        digit_shaped: Check whether a box has the aspect ratio of a seven-segment digit

        :param box (tuple): box as (x, y, w, h)
        :return: True if the width is between 10% and 1/1.4 of the height (bool)
        """
        return 0.1 * box[3] <= box[2] <= box[3] / 1.4

    whole = [box for box in blobs if digit_shaped(box)]
    if not whole:
        return []
    
    blobs = merge_fragments(blobs, max(box[3] for box in whole), mask.shape)
    candidates = [box for box in blobs if digit_shaped(box)]

    # Drop small specks: the pulse digits are the smallest real ones (~60% of
    # the systolic height)
    max_height = max(box[3] for box in candidates)
    digits = [box for box in candidates if box[3] >= 0.4 * max_height]

    rows = [clean_row(row, mask.shape) for row in group_rows(digits)]
    return [row for row in rows if row]


def crop_box(img, box, pad_frac=0.01):
    """
    crop_box: Crop (x, y, w, h) with a small padding, clipped to the image

    :param img (np.ndarray): source image
    :param box (tuple): box as (x, y, w, h)
    :param pad_frac (float): padding as a fraction of the box's longer side
    :return: the cropped region (np.ndarray)
    """
    box_x, box_y, box_width, box_height = box
    pad_px = int(round(pad_frac * max(box_width, box_height)))
    img_height, img_width = img.shape[:2]
    
    left, top = max(0, box_x - pad_px), max(0, box_y - pad_px)
    right, bottom = min(img_width, box_x + box_width + pad_px), min(img_height, box_y + box_height + pad_px)

    return img[top:bottom, left:right]


def draw_lcd_debug(img, rows):
    """
    draw_lcd_debug: Draw the detected digit boxes and their output names on the LCD image

    :param img (np.ndarray): BGR LCD image
    :param rows (list): rows of boxes (x, y, w, h), as from find_lcd_digits
    :return: annotated copy of the image (np.ndarray)
    """
    annotated = img.copy()
    # Line thickness scaled to the image width
    thickness = max(2, img.shape[1] // 300)
    digit_count = 0

    for row in rows:
        for box_x, box_y, box_width, box_height in row:
            digit_count += 1
            cv2.rectangle(
                annotated,
                (int(box_x), int(box_y)),
                (int(box_x + box_width), int(box_y + box_height)),
                (0, 0, 255),        # red box
                thickness,
            )
            cv2.putText(
                annotated,
                f"d{digit_count}",                          # output file name of this digit
                (int(box_x), int(box_y) + 3 * thickness + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                img.shape[1] / 600,                         # font scale relative to the image width
                (0, 160, 0),                                # green text
                thickness,
            )
    return annotated


# --------------------------------------------------------------------------- #
# Thermometer
# --------------------------------------------------------------------------- #

def fit_liquid_line(response, max_tilt_deg=6.0):
    """
    fit_liquid_line: Find the line x = slope*y + intercept along which the liquid
    response is strongest

    A brute-force search over small tilts (a Hough transform restricted to
    near-vertical lines), so a plate Task 1 left slightly rotated still works

    :param response (np.ndarray): liquid response image, from liquid_response
    :param max_tilt_deg (float): largest tilt from vertical tried, in degrees
    :return: (slope, intercept) of the line
    """
    height, width = response.shape
    # Sum over row blocks first: much faster, and the line barely moves
    # sideways within one block.
    block_size = max(1, height // 200)
    num_blocks = height // block_size

    blocks = response[:num_blocks * block_size].reshape(num_blocks, block_size, width).mean(axis=1)
    block_centres_y = (np.arange(num_blocks) + 0.5) * block_size

    centre_y = height / 2

    # Candidate line positions at the centre row: the middle 60% of the width
    x_min, x_max = int(0.2 * width), int(0.8 * width)
    offsets = np.arange(x_min, x_max)
    max_slope = np.tan(np.radians(max_tilt_deg))
    best = (-1.0, 0.0, float(offsets[0]))

    for slope in np.linspace(-max_slope, max_slope, 49):
        # Column of each candidate line in every block
        columns = np.round(offsets[None, :] + slope * (block_centres_y[:, None] - centre_y)).astype(int)
        valid = (columns >= 0) & (columns < width)
        values = np.where(valid, blocks[np.arange(num_blocks)[:, None], np.clip(columns, 0, width - 1)], 0.0)

        # Liquid is a 2-4 px line, so sum a 3-column window around each offset
        score = values.sum(axis=0)
        score = np.convolve(score, np.ones(3), mode="same")
        best_idx = int(np.argmax(score))

        if score[best_idx] > best[0]:
            best = (score[best_idx], slope, float(offsets[best_idx]))
    _, slope, x_centre = best

    # Convert the line from "x at the centre row" to an intercept at row 0
    return slope, x_centre - slope * centre_y


def liquid_response(img):
    """
    liquid_response: Compute a per-pixel "thin coloured vertical line" strength,
    high on the liquid

    :param img (np.ndarray): BGR thermometer image
    :return: response image (np.ndarray, float32)
    """
    height, width = img.shape[:2]
    # Chroma (max - min channel) rather than HSV saturation: saturation
    # explodes on noisy dark pixels, which would make the black scale print
    # look "coloured"
    chroma = (img.max(axis=2).astype(np.int16) - img.min(axis=2)).astype(np.uint8)
    chroma = cv2.GaussianBlur(chroma, (0, 0), 1.0)

    # The liquid is a thin, coloured vertical line; the scale print is black
    # (no chroma). After reversing the chroma image (as in prac02) the liquid
    # is a thin dark line, and a horizontal black-hat (closing minus image)
    # keeps only such narrow structures, i.e. the liquid
    kernel_width = max(5, int(round(width * 0.05)))
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    response = cv2.morphologyEx(255 - chroma, cv2.MORPH_BLACKHAT, element).astype(np.float32)

    return cv2.GaussianBlur(
        response,
        (1, 0),
        sigmaX=0.1,                         # no horizontal blur, so the line is not widened
        sigmaY=max(1.0, height * 0.004),    # vertical blur along the line suppresses pixel noise
    )


def find_liquid_top(response, line, y_range=None):
    """
    find_liquid_top: Find the row of the top of the liquid column along the fitted line

    y_range limits the search to the printed scale. The coloured bulb at the
    bottom of the tube responds far more strongly than the thin column, and if
    it is left in, the threshold below splits "bulb vs column" instead of
    "column vs empty tube"

    :param response (np.ndarray): liquid response image
    :param line (tuple): (slope, intercept) of the liquid line
    :param y_range (tuple): (start, end) rows to search, or None for the whole image
    :return: row of the liquid top (int), or None if not found
    """
    height, width = response.shape
    slope, intercept = line
    y_start, y_end = (0, height) if y_range is None else (max(0, int(y_range[0])), min(height, int(y_range[1])))
    if y_end - y_start < 10:
        return None

    # Sample the response along the fitted line, a few pixels either side
    half_width = max(2, int(round(width * 0.006)))
    rows = np.arange(y_start, y_end)
    centre_cols = np.clip(np.round(slope * rows + intercept).astype(int), half_width, width - half_width - 1)

    profile = np.array([response[row, col - half_width:col + half_width + 1].mean()
                        for row, col in zip(rows, centre_cols)])
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (0, 0), max(1, height * 0.003)).ravel()

    # Otsu on the profile values splits "empty tube" from "liquid", so a
    # constant offset from noise or lighting does not matter
    scaled = cv2.normalize(profile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Clip rare very high values (glare, any bit of bulb left in the range) so
    # they do not drag the threshold up
    scaled = np.minimum(scaled, np.uint8(np.percentile(scaled, 95)))
    threshold, _ = cv2.threshold(scaled.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    is_liquid = scaled > threshold

    # Bridge short dropouts (glare, tick shadows) before looking for the run
    gap_px = max(3, int(round(height * 0.01)))
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap_px))
    is_liquid = cv2.morphologyEx(is_liquid.astype(np.uint8).reshape(-1, 1), cv2.MORPH_CLOSE, element).ravel() > 0

    # The liquid is the longest "on" run; its top is the reading
    best_len, best_top, run_start = 0, None, None
    # A trailing False closes a run that reaches the last row
    for row, value in enumerate(np.append(is_liquid, False)):
        if value and run_start is None:
            run_start = row
        elif not value and run_start is not None:
            if row - run_start > best_len:
                best_len, best_top = row - run_start, run_start
            run_start = None
    return None if best_top is None else y_start + best_top


def find_scale_labels(mask, liquid_x):
    """
    find_scale_labels: Find the vertical centres of the numbered labels on the left
    (degC) scale

    :param mask (np.ndarray): dark print mask
    :param liquid_x (float): column of the liquid line
    :return: label centre rows, top to bottom (list of float)
    """
    height, width = mask.shape
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Label digits are compact, glyph-sized blobs left of the tube. Tick marks
    # are thin and wide, so the aspect test removes them. A blob is kept only
    # if it passes the size, aspect and left-of-tube tests
    glyphs = []
    for label in range(1, num_labels):
        box_x, box_y, box_width, box_height, _ = stats[label]
        if (0.04 * width <= box_height <= 0.15 * width
                and box_width <= 1.2 * box_height
                and box_x + box_width <= liquid_x - 0.08 * width):
            glyphs.append((centroids[label][1], box_height))
    if not glyphs:
        return []

    # Glyphs of one label ("-20") share a row; merge them
    glyphs.sort()
    rows = []

    for centre_y, box_height in glyphs:
        if rows and centre_y - rows[-1][-1][0] < 0.5 * box_height:
            rows[-1].append((centre_y, box_height))
        else:
            rows.append([(centre_y, box_height)])
    centres = [float(np.mean([glyph[0] for glyph in row])) for row in rows]

    # The labels are evenly spaced. Anything at either end with an unusual gap
    # (e.g. the "degC" heading) is dropped
    while len(centres) >= 3:
        gaps = np.diff(centres)
        median_gap = np.median(gaps)
        if abs(gaps[0] - median_gap) > 0.3 * median_gap:
            centres.pop(0)
        elif abs(gaps[-1] - median_gap) > 0.3 * median_gap:
            centres.pop()
        else:
            break
    return centres


def find_scale_ticks(mask, liquid_x, y_range):
    """
    find_scale_ticks: Find the vertical positions of the tick marks on the left (degC) scale

    The ticks form a band of short horizontal bars just left of the tube. The
    band is the run of mostly-dark columns nearest the tube (the tube itself is
    print-free); within it, each tick is a run of dark rows

    :param mask (np.ndarray): dark print mask
    :param liquid_x (float): column of the liquid line
    :param y_range (tuple): (start, end) rows to search
    :return: (tick rows (list of float), (band_left, band_right) columns of the tick band)
    """
    height, width = mask.shape
    y_start, y_end = max(0, int(y_range[0])), min(height, int(y_range[1]))

    # Search the 20% of the width just left of the tube
    x_min = max(0, int(liquid_x - 0.2 * width))
    x_max = int(liquid_x)

    if y_end - y_start < 10 or x_max - x_min < 5:
        return [], (x_min, x_min)

    # x distribution (dark pixels per column, as in prac04) as a fraction
    region = mask[y_start:y_end, x_min:x_max] > 0
    col_density = np.sum(region, axis=0) / region.shape[0]
    threshold = 0.5 * col_density.max()

    # Walk left from the tube: skip the empty tube, then take the dark band
    col = len(col_density) - 1
    while col >= 0 and col_density[col] < threshold:
        col -= 1
    band_right = col + 1
    while col >= 0 and col_density[col] >= threshold:
        col -= 1
    band_left = col + 1
    if band_right - band_left < 2:
        return [], (x_min, x_min)

    # y distribution of the band: a tick row is mostly dark
    band = mask[y_start:y_end, x_min + band_left:x_min + band_right] > 0
    row_density = np.sum(band, axis=1) / band.shape[1]
    is_tick = np.append(row_density > 0.5, False)
    ticks, run_start = [], None
    # Each run of tick rows gives one tick at the run's middle row
    for row, value in enumerate(is_tick):
        if value and run_start is None:
            run_start = row
        elif not value and run_start is not None:
            ticks.append(y_start + (run_start + row - 1) / 2)
            run_start = None
    return ticks, (x_min + band_left, x_min + band_right)


def analyse_thermometer(img):
    """
    analyse_thermometer: Locate the liquid top and the scale around it, and choose the
    crop rows

    :param img (np.ndarray): BGR thermometer image
    :return: dict with the liquid "line", liquid "top" row, "labels" and "ticks" rows,
             label "spacing", "tick_band" columns and "crop" rows (y0, y1)
    """
    height, width = img.shape[:2]
    response = liquid_response(img)
    line = fit_liquid_line(response)
    # Column of the liquid line at the centre row
    liquid_x = line[0] * height / 2 + line[1]

    mask = dark_print_mask(img)
    labels = find_scale_labels(mask, liquid_x)
    # Pixels between neighbouring labels (10 degC)
    spacing = float(np.median(np.diff(labels))) if len(labels) >= 2 else None

    # The reading lies on the printed scale, so look for the liquid top only
    # between the end labels (half a label step of slack either side). This
    # keeps the coloured bulb below the scale out of the search
    scale_range = (labels[0] - 0.5 * spacing, labels[-1] + 0.5 * spacing) if spacing else None
    top = find_liquid_top(response, line, scale_range)

    if top is None and scale_range is not None:
        top = find_liquid_top(response, line)
    if top is None:
        raise RuntimeError("liquid column not found")

    if spacing:
        # The pair of labels bracketing the liquid top (clamped to the ends)
        label_idx = int(np.searchsorted(labels, top))
        label_idx = min(max(label_idx, 1), len(labels) - 1)
        above, below = labels[label_idx - 1], labels[label_idx]
        # Centre the crop on the liquid top, and make it tall enough to show
        # both bracketing labels in full (a label is ~0.3 steps tall)
        margin = 0.3 * spacing
        half_height = max(top - (above - margin), (below + margin) - top)
    else:
        # No readable scale: fall back to a window centred on the liquid top
        half_height = 0.3 * width
    crop_top = int(max(0, round(top - half_height)))
    crop_bottom = int(min(height, round(top + half_height)))

    ticks, band = find_scale_ticks(mask, liquid_x, (crop_top, crop_bottom))
    return {"line": line, "top": int(top), "labels": labels, "spacing": spacing,
            "ticks": ticks, "tick_band": band, "crop": (crop_top, crop_bottom)}


def draw_thermo_debug(img, info):
    """
    draw_thermo_debug: Draw the liquid line, labels, ticks, liquid top and crop on the
    thermometer image

    :param img (np.ndarray): BGR thermometer image
    :param info (dict): result of analyse_thermometer
    :return: annotated copy of the image (np.ndarray)
    """
    annotated = img.copy()
    height, width = img.shape[:2]
    # Line thickness scaled to the image width
    thickness = max(2, width // 300)
    slope, intercept = info["line"]
    # Liquid line in blue
    cv2.line(annotated, (int(round(intercept)), 0), (int(round(slope * height + intercept)), height), (255, 0, 0), 1)

    # Scale labels in green
    for label_y in info["labels"]:
        cv2.line(annotated, (0, int(label_y)), (int(width * 0.15), int(label_y)), (0, 160, 0), thickness)

    # Ticks in magenta, across the tick band
    band_left, band_right = info["tick_band"]
    for tick_y in info["ticks"]:
        cv2.line(annotated, (band_left, int(tick_y)), (band_right, int(tick_y)), (255, 0, 255), 1)
    # Liquid top in red
    top = info["top"]
    cv2.line(annotated, (0, top), (width, top), (0, 0, 255), thickness)
    # Crop rows in orange
    crop_top, crop_bottom = info["crop"]
    cv2.rectangle(annotated, (0, crop_top), (width - 1, crop_bottom - 1), (0, 200, 255), thickness)

    return annotated


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def is_thermometer(path, img):
    """
    is_thermometer: Decide the device type from the file name (lcdX / thermoX), else
    from the image shape

    :param path (str): image path
    :param img (np.ndarray): the image
    :return: True for a thermometer, False for a BPM LCD (bool)
    """
    name = os.path.basename(path).lower()
    if name.startswith("thermo"):
        return True
    if name.startswith("lcd"):
        return False
    height, width = img.shape[:2]
    return height / width >= THERMO_MIN_ASPECT


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


def read_settings(config):
    """
    read_settings: Read the Task 2 settings from config.txt, falling back to the defaults

    :param config (dict): key/value pairs from config.txt, or None
    :return: dict with "debug"
    """
    config = config or {}
    return {"debug": int(config.get("task2_debug", DEFAULT_DEBUG)) != 0}


def process_lcd(img, out_dir):
    """
    process_lcd: Segment the digits of one LCD image and save each as dN.png

    :param img (np.ndarray): BGR LCD image
    :param out_dir (str): folder for the digit crops
    :return: debug image with the digit boxes (np.ndarray)
    """
    rows = find_lcd_digits(img)
    sizes = ", ".join(str(len(row)) for row in rows)
    print(f"  bpm: {sum(len(row) for row in rows)} digit(s) in {len(rows)} row(s) [{sizes}]")

    if len(rows) != 3:
        print("  warning: expected 3 rows (SYS, DIA, PULSE)")
    
    # Flatten the rows into reading order and save d1.png, d2.png, ...
    boxes = [box for row in rows for box in row]

    for digit_idx, box in enumerate(boxes, start=1):
        save_output(os.path.join(out_dir, f"d{digit_idx}.png"), crop_box(img, box), output_type='image')
    return draw_lcd_debug(img, rows)


def process_thermo(img, out_dir):
    """
    process_thermo: Crop one thermometer image around the liquid top and save it as t.png

    :param img (np.ndarray): BGR thermometer image
    :param out_dir (str): folder for the crop
    :return: debug image with the liquid top, labels, ticks and crop (np.ndarray)
    """
    height = img.shape[0]
    info = analyse_thermometer(img)
    crop_top, crop_bottom = info["crop"]
    print(f"  thermo: liquid top at row {info['top']} ({100 * info['top'] / height:.1f}% down)")

    if info["spacing"]:
        print(f"  {len(info['labels'])} scale labels, {info['spacing']:.1f} px apart (10 degC)")
    else:
        print("  warning: scale labels not found, cropping around the liquid top only")

    if len(info["ticks"]) >= 2:
        tick_step = float(np.median(np.diff(info["ticks"])))
        print(f"  {len(info['ticks'])} tick marks in the crop, {tick_step:.1f} px apart (1 degC)")

    print(f"  crop rows {crop_top}-{crop_bottom}")
    save_output(os.path.join(out_dir, "t.png"), img[crop_top:crop_bottom, :], output_type='image')

    return draw_thermo_debug(img, info)


def run_task2(image_path, config):
    """
    run_task2: Segment every rectified image and save the crops to output/task2/<name>/

    :param image_path (str): a single image file or a folder of images
    :param config (dict): key/value pairs from config.txt
    :return: None
    """
    print("=============================================================")
    print("TASK 2 - Segmenting images and extracting crops")
    print("=============================================================")

    settings = read_settings(config)
    images = list_images(image_path)
    print(f"Note: {len(images)} image(s) in {image_path} ({settings})")

    for path in images:
        img = cv2.imread(path)
        if img is None:
            print(f"Could not read image: {path}")
        else:
            # Only a readable image is segmented
            name = os.path.splitext(os.path.basename(path))[0]
            out_dir = os.path.join(OUTPUT_DIR, name)
            print(f"[{os.path.basename(path)}]:")

            try:
                if is_thermometer(path, img):
                    debug_image = process_thermo(img, out_dir)
                else:
                    debug_image = process_lcd(img, out_dir)
                if settings["debug"]:
                    save_output(os.path.join(OUTPUT_DIR, "debug", f"debug_{name}.png"), debug_image,
                                output_type='image')
                print("------")
            except Exception as error:
                # One bad image must not stop the rest of the batch
                print(f"Task 2 failed on {path}: {error}")
