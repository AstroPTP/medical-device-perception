# Medical Device Perception

**Reading blood-pressure monitors and analogue thermometers from ordinary phone photos.**

A four-stage computer-vision pipeline that takes a raw photo, finds the medical device in it, straightens its display, segments the reading, and returns the value as text:

```
img2.jpg  ->  bpm 135,93,65
img1.jpg  ->  thermo 18
```

It combines a fine-tuned **YOLOv8-pose** detector, **classical OpenCV geometry** for sub-pixel-level corner refinement and segmentation, and a **custom PyTorch CNN** trained mostly on **synthetic data** I generated myself.

> **University assignment:** COMP3007 Machine Perception, Curtin University, Semester 2 2026.
> Python · PyTorch · Ultralytics YOLOv8 · OpenCV · NumPy · scikit-learn

---

## About this project

This is my solution to the major assignment for **COMP3007 Machine Perception** at **Curtin University**.

| | |
|---|---|
| **Unit** | COMP3007 Machine Perception |
| **Institution** | Curtin University, Perth, Western Australia |
| **Unit Coordinator** | Sonny Pham |
| **Semester** | Semester 2, 2026 |
| **Time frame** | August to October 2026 (due Week 12, 9 October 2026) |
| **Weight** | 40% of the unit mark |
| **Type** | Individual assignment: code, a 12-page technical report, a video demo and a live interview on an unseen test set |

**The brief.** Build a perception module for a health app that photographs medical devices and digitises their readings. Two devices are covered: an Omron-style **blood pressure monitor** with a seven-segment LCD (systolic, diastolic and pulse rows), and an analogue **wall thermometer** (about −20 °C to 50 °C, 1 °C ticks). The hidden test set has five photos: an upright BPM, a rotated BPM, a BPM with perspective distortion, a slightly rotated thermometer, and a negative image with no device.

**Constraints set by the unit:**

- Python 3.10, approved libraries only, inside a provided assignment framework (`assignment.py`)
- **No pre-made OCR** (Tesseract, PaddleOCR, EasyOCR and similar are banned). The digit classifier had to be trained or fine-tuned by me
- Each task must finish within 5 minutes on a lab machine, with no user input or interactive plots
- Training data: unlabelled photos supplied by the unit, which I annotated myself in Roboflow
- Thermometer readings had to be within ±1 °C of the true value for full marks

The four tasks below (detection, segmentation, recognition, full pipeline) follow the assignment's structure, and each one was assessed separately.

---

## Highlights

- **Hybrid learned + classical design.** The neural network does what it is good at (finding and roughly locating the device); classical CV does what it is good at (pixel-accurate edges and lines). Each stage hands the next one a cleaner, simpler problem.
- **Keypoint detection for orientation.** Instead of a plain bounding box, the detector predicts the four corners of the reading region *in the device's own reading order*, so one perspective warp fixes rotation, tilt and perspective at once, even for upside-down photos.
- **Hough-style corner refinement.** YOLO's corners can be tens of pixels off. Each LCD border is re-found by searching small shifts and rotations around the predicted line for the strongest dark-to-bright edge, and the corners are rebuilt from the line intersections. For thermometers, the glass tube's direction gives the true rotation.
- **Synthetic training data.** I had only ~200 real digit crops and none of them showed a `0`. To cover that, I wrote a renderer for seven-segment digits (random segment shapes, unlit "ghost" segments, lighting, blur, noise, rotation, crop jitter) and for printed scale numerals (multiple fonts, condensing, neighbouring glyphs).
- **Robust thermometer decoding.** The printed scale labels are decoded *jointly*: every possible value for the top label is scored against the network's reads of all labels, so one misread glyph gets outvoted. The final temperature is interpolated between the 1 °C ticks either side of the liquid top.
- **Leak-free evaluation.** Training photos that are byte-identical to validation images (checked by file hash) are excluded, so the validation digits are a genuinely held-out test set.

## Results

| Component | Metric | Result |
|---|---|---|
| Device detector (YOLOv8n-pose, 150 epochs @ 960 px) | Keypoint mAP@50 | **0.995** |
| | Keypoint mAP@50–95 | **0.99** |
| | Box precision / recall | 0.93 / 0.95 |
| End-to-end pipeline (held-out validation photos) | BPM readings (SYS, DIA, PULSE) | **3 / 3 exactly correct** |
| | Negative images (no device) | correctly produce no output |

Example end-to-end outputs on the validation set:

| Photo | Output |
|---|---|
| img2 | `bpm 135,93,65` |
| img4 | `bpm 134,88,75` |
| img5 | `bpm 134,89,72` |
| img1 | `thermo 18` |

<!-- Suggested: add a pipeline figure here, e.g.
![Pipeline](docs/images/pipeline.png)
-->

## How it works

```
 photo ──► Task 1 ──► Task 2 ──► Task 3 ──► Task 4
          detect &   segment    recognise   one-line
          rectify    reading    reading     result
```

### Task 1: Detect and rectify ([task1.py](task1.py))

1. **Learned detection.** A YOLOv8n-pose model ([train_task1.py](train_task1.py)) classifies the device as `bpm` or `thermo` and predicts four keypoints: the corners of the LCD panel, or of the printed thermometer scale with the 50 °C end on top.
2. **Classical refinement.**
   - *BPM:* the LCD is darker than its white bezel, so each border is a long straight edge. A small search over offsets and angles near each predicted border finds the strongest edge; the refined corners are the intersections of the four lines.
   - *Thermometer:* the scale has no physical border, but the glass tube runs exactly along it. The tube's direction gives the true rotation, and the predicted corners give the extent.
   - Outlier corners are repaired and implausible refinements (too large a move or area change) are rejected, falling back to the model's prediction.
3. **Perspective warp** to an upright rectangle. If no device is found, the image produces no output.

### Task 2: Segment the reading ([task2.py](task2.py))

Pure OpenCV. All thresholds are relative to image size, so nothing depends on input resolution.

- **BPM:** a background-normalised dark-segment mask, connected components, merging of broken seven-segment fragments, grouping into rows, and removal of border artefacts. Each digit is cropped in reading order (systolic, diastolic, pulse).
- **Thermometer:** a line fit to the liquid column (tolerant to slight tilt), detection of the liquid top, and a crop centred on it that reaches the numbered labels above and below.

### Task 3: Recognise ([task3.py](task3.py))

- **DigitCNN:** a small VGG-style network (~0.5 M parameters, 48×32 grayscale input) that runs quickly on a CPU. Digits are resized with their aspect ratio kept, because a narrow `1` is part of what makes it a `1`.
- Two copies are trained ([train_task3.py](train_task3.py)): `lcd` for seven-segment digits (real and synthetic) and `print` for the thermometer's printed numerals (synthetic only).
- **Thermometer reading:** find the 1 °C ticks and their spacing, read the labels with the `print` network, decode them jointly under the constraint that they go down in steps of 10 °C, then interpolate the liquid-top position to a temperature.

### Task 4: End-to-end ([task4.py](task4.py))

Chains Tasks 1 to 3 in memory, with no intermediate files and each model loaded once per batch, and writes one line per photo:

```
bpm SYS,DIA,PULSE
thermo TEMP
```

## Getting started

### Requirements

Python 3.10. Main dependencies: `torch`, `ultralytics`, `opencv-python`, `numpy`, `scikit-learn`, `matplotlib`, `Pillow`.

```bash
pip install torch ultralytics opencv-python numpy scikit-learn matplotlib pillow
```

Or recreate the full Conda environment:

```bash
conda env create -f comp3007.yml
```

Pre-trained weights are included in [data/](data/) (`task1_pose.pt`, `task3_digits.pt`), so inference works without retraining.

### Run inference

```bash
python assignment.py task1 path/to/photos
```

```bash
python assignment.py task4 path/to/photos
```

Each task reads its input folder and writes to `output/taskN/`. Tasks 2 and 3 expect the previous task's output as input; Task 4 runs the whole pipeline from raw photos.

Optional settings in `config.txt`:

| Key | Default | Meaning |
|---|---|---|
| `task1_conf` | 0.5 | Minimum detection confidence |
| `task1_imgsz` | 960 | Detector inference size |
| `task1_pad` | 0.0 | Grow the crop outward by this fraction |
| `task1_refine` | 1 | `0` disables classical corner refinement |

### Retrain the models

Both training scripts work with your own data. Every path has a default inside the project folder and can be changed on the command line. The training data isn't included in this repository.

**Detector (`train_task1.py`).** Expects a YOLOv8 **pose** dataset (a Roboflow "YOLOv8" export works as is). Each label is one box (`bpm` or `thermo`) plus **4 keypoints**: the reading-region corners in the order top-left, top-right, bottom-right, bottom-left, as seen when reading the device.

```
dataset/                 # default location, or pass --data
├── data.yaml            # nc, names, kpt_shape: [4, 3]; split paths optional
├── train/images, train/labels
├── valid/images, valid/labels
└── test/images, test/labels   (optional)
```

```bash
python train_task1.py --data path/to/your/dataset --epochs 150 --imgsz 960
```

The script writes a corrected copy of `data.yaml` to `runs/`, with absolute paths and a fixed `flip_idx` that swaps left and right corners under horizontal flips. Split paths are read from `data.yaml`; Roboflow's `../train/images` style is handled.

**Digit classifier (`train_task3.py`).** Trains mostly on synthetic digits, so it runs even with no photos. Real BPM photos are optional and improve the LCD network:

| Option | Default | What it is |
|---|---|---|
| `--photos` | `./unlabelled_data` | Folder of BPM photos (`.jpg`, `.png`, ...) |
| `--readings` | `<photos>/readings.csv` | Labels for those photos, columns `photo,reading`, e.g. `IMG_0001.jpg,"118,78,70"` |
| `--validation` | `./validation` | Held-out set: `*/imgX.jpg` photos plus `task3/lcdX/dN.png` digit crops |
| `--validation-labels` | `<validation>/task3_labels.csv` | Columns `folder,digits`, e.g. `lcd2,1359365` (all digits in reading order) |

```bash
python train_task3.py --photos path/to/bpm_photos --validation path/to/validation
```

If no label files are found, the script uses the built-in labels for the original dataset. Photos that are byte-identical to a validation image are never trained on. Confusion matrices for the synthetic hold-out data and the real validation digits are saved under `runs/task3/`.

## Project structure

```
├── assignment.py      # CLI entry point: python assignment.py <task> <input_dir>
├── task1.py           # Device detection, corner refinement, rectification
├── task2.py           # Digit / thermometer-scale segmentation
├── task3.py           # DigitCNN + thermometer reading
├── task4.py           # End-to-end pipeline
├── train_task1.py     # YOLOv8-pose fine-tuning
├── train_task3.py     # Synthetic data generation + CNN training
├── data/              # Trained weights
└── comp3007.yml       # Conda environment
```

## What I learned

- **Know when not to use deep learning.** Once Task 1 hands over an upright, tightly cropped image, classical segmentation is faster, easier to explain and easier to debug than another network.
- **Loss design matters more than epochs.** I tried tightening the keypoint OKS sigma from Ultralytics' default (0.25) to 0.05 for sharper corners. It made results *worse*: the loss saturated for far-off corners, so they got almost no gradient, and the thermometer corners collapsed. Keeping the loose sigma and refining classically worked better.
- **Synthetic data covers rare and missing classes.** A procedural renderer filled gaps (such as the digit `0`) that the real data could never cover.
- **Use structure as a prior.** Reading thermometer labels jointly, using the known 10 °C spacing, is far more robust than trusting each digit on its own.

## Author

**Thanh Phu Phan** · [GitHub @AstroPTP](https://github.com/AstroPTP)
