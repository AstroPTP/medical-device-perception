

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
Task 4: the complete pipeline, from a photo to a one-line reading.

For each input photo imgX.jpg:

  1. Task 1: detect and classify the device, and warp its reading region to an
     upright rectangle. No device -> no output file for this image.
  2. Task 2: segment the region.
       - BPM:    digit boxes grouped into rows (SYS, DIA, PULSE).
       - Thermo: a crop centred on the liquid top with the labels around it.
  3. Task 3: recognise the reading.
       - BPM:    each digit with the seven-segment CNN; each row joined into
                 a number.
       - Thermo: temperature interpolated between the scale ticks.
  4. Write output/task4/imgX.txt with one line:
       bpm SYS,DIA,PULSE    e.g. "bpm 118,78,70"
       thermo TEMP          e.g. "thermo 37"

Everything stays in memory: the stages are called as functions of the other
task modules, so no intermediate files are needed and each model is loaded
only once for the whole batch.

Settings in config.txt: the Task 1 settings (task1_conf, task1_refine, ...)
also apply here. Set "task4_debug: 1" to also save the intermediate images
(reading region, digit or scale crop) under output/task4/debug/imgX/.
"""

import os

import cv2
import numpy as np

import task1
import task2
import task3

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output", "task4")

DEFAULT_DEBUG = 0      # 1 = also save intermediate images per input

# The BPM display shows systolic, diastolic and pulse rows.
BPM_ROWS = 3


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
# Blood pressure monitor
# --------------------------------------------------------------------------- #

def pick_reading_rows(rows):
    """
    Keep the SYS, DIA and PULSE rows if Task 2 found extra ones.

    A stray row (e.g. a sliver of an icon that passed the digit tests) has
    fewer and smaller boxes than the real readings, so the rows with the most
    digits (then the tallest) are kept, in their original top-to-bottom order.
    """
    if len(rows) <= BPM_ROWS:
        return rows

    def strength(i):
        return len(rows[i]), float(np.median([b[3] for b in rows[i]]))

    keep = sorted(sorted(range(len(rows)), key=strength, reverse=True)[:BPM_ROWS])
    return [rows[i] for i in keep]


def read_bpm(region, debug_dir=None):
    """
    Read SYS, DIA and PULSE from a rectified LCD image.

    Returns (list of row values as strings, list of per-digit confidences).
    """
    rows = task2.find_lcd_digits(region)
    sizes = ", ".join(str(len(r)) for r in rows)
    print(f"  segmentation: {sum(len(r) for r in rows)} digit(s) in {len(rows)} row(s) [{sizes}]")
    rows = pick_reading_rows(rows)
    if len(rows) != BPM_ROWS:
        print(f"  warning: expected {BPM_ROWS} rows (SYS, DIA, PULSE), found {len(rows)}")

    crops = [task2.crop_box(region, box) for row in rows for box in row]
    results = task3.recognise_digits(crops)

    values, confs, k = [], [], 0
    for row in rows:
        row_results = results[k:k + len(row)]
        k += len(row)
        values.append("".join(str(d) for d, _ in row_results))
        confs.extend(c for _, c in row_results)

    if debug_dir:
        save_output(os.path.join(debug_dir, "digits.png"), task2.draw_lcd_debug(region, rows),
                    output_type='image')
        for i, crop in enumerate(crops, start=1):
            save_output(os.path.join(debug_dir, f"d{i}.png"), crop, output_type='image')
    return values, confs


# --------------------------------------------------------------------------- #
# Thermometer
# --------------------------------------------------------------------------- #

def read_thermo(region, debug_dir=None):
    """
    Temperature (float degC) from a rectified thermometer scale image.

    Task 2 crops the scale around the liquid top, then Task 3 reads that crop.
    If either step fails on the crop, Task 3 is tried on the whole region
    (it does not need the crop, it is just cleaner with it).
    """
    try:
        seg = task2.analyse_thermometer(region)
        y0, y1 = seg["crop"]
        crop = region[y0:y1, :]
        print(f"  segmentation: liquid top at row {seg['top']}, crop rows {y0}-{y1}")
        info = task3.read_thermometer(crop)
    except RuntimeError as e:
        print(f"  segmented crop failed ({e}); reading the whole scale instead")
        crop = region
        info = task3.read_thermometer(crop)

    labels = ", ".join(str(v) for v in info["values"])
    print(f"  labels read: {labels} (confidence {info['label_conf']:.2f}), "
          f"{len(info['ticks'])} ticks, {info['deg_px']:.1f} px per degC")
    print(f"  temperature: {info['temp']:.2f} degC (from {info['method']})")

    if debug_dir:
        save_output(os.path.join(debug_dir, "t.png"), crop, output_type='image')
        save_output(os.path.join(debug_dir, "reading.png"), task3.draw_thermo_debug(crop, info),
                    output_type='image')
    return info["temp"]


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def read_settings(config):
    """Task 1 settings (detection) plus the Task 4 debug switch."""
    settings = task1.read_settings(config)
    settings["debug"] = int((config or {}).get("task4_debug", DEFAULT_DEBUG)) != 0
    return settings


def process_image(img, settings, debug_dir=None):
    """
    Full pipeline for one photo.

    Returns the output line ("bpm 118,78,70" / "thermo 37"), or None if no
    device was detected.
    """
    found = task1.extract_reading_region(img, conf=settings["conf"], imgsz=settings["imgsz"],
                                         pad=settings["pad"], refine=settings["refine"])
    if found is None:
        return None
    class_name, region, confidence = found
    print(f"  detection: {class_name} (conf {confidence:.2f}), "
          f"region {region.shape[1]}x{region.shape[0]}")
    if debug_dir:
        prefix = task1.PREFIX.get(class_name, class_name)
        save_output(os.path.join(debug_dir, f"{prefix}.png"), region, output_type='image')

    if class_name == "bpm":
        values, confs = read_bpm(region, debug_dir)
        if confs:
            print(f"  digits: {' '.join(values)} (lowest confidence {min(confs):.2f})")
        return f"bpm {','.join(values)}"
    if class_name == "thermo":
        temp = read_thermo(region, debug_dir)
        return f"thermo {task3.round_half_up(temp)}"
    raise RuntimeError(f"unknown device class: {class_name}")


def run_task4(image_path, config):
    settings = read_settings(config)
    images = task1.list_images(image_path)
    print(f"Task 4: {len(images)} image(s) in {image_path} ({settings})")

    for path in images:
        img = cv2.imread(path)
        if img is None:
            print(f"Could not read image: {path}")
            continue

        name = os.path.splitext(os.path.basename(path))[0]
        debug_dir = os.path.join(OUTPUT_DIR, "debug", name) if settings["debug"] else None
        print(f"{os.path.basename(path)}:")
        try:
            line = process_image(img, settings, debug_dir)
            if line is None:
                print("  no device detected, no output")
                continue
            print(f"  result: {line}")
            save_output(os.path.join(OUTPUT_DIR, f"{name}.txt"), line, output_type='txt')
        except Exception as e:
            # One bad image must not stop the rest of the batch.
            print(f"Task 4 failed on {path}: {e}")
