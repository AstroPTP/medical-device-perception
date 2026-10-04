
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
Train the Task 1 device detector: YOLOv8n-pose fine-tuned on the Roboflow
export "medical-device-perception" (YOLOv8 pose format).

Each label is one device box (class bpm or thermo) plus four keypoints, the
corners of the reading region in the device's own reading orientation:
0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left.
  - bpm    : corners of the LCD panel
  - thermo : corners of the thermometer plate

The best checkpoint is copied to data/task1_pose.pt, which task1.py loads.

Usage (inside the comp3007 conda environment):
    python train_task1.py
    python train_task1.py --data ../dataset/medical-device-perception.v3i.yolov8 --epochs 150
"""

import argparse
import os
import shutil

import yaml
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "dataset", "medical-device-perception.v3i.yolov8")
WEIGHTS_OUT = os.path.join(HERE, "data", "task1_pose.pt")


def write_dataset_yaml(dataset_dir, out_path):
    """
    Write a corrected copy of the Roboflow data.yaml.

    The export sets flip_idx to [0, 1, 2, 3], which is wrong: a horizontal
    flip turns the top-left corner into the top-right one (and bottom-right
    into bottom-left), so the keypoint indices must be swapped as [1, 0, 3, 2].
    Without this, flip augmentation teaches the model mirrored corner orders.
    Absolute paths are also used so training works from any directory.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    with open(os.path.join(dataset_dir, "data.yaml")) as f:
        src = yaml.safe_load(f)

    cfg = {
        "path": dataset_dir,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "kpt_shape": [4, 3],
        "flip_idx": [1, 0, 3, 2],
        "nc": src["nc"],
        "names": src["names"],
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Train the Task 1 YOLOv8n-pose detector")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Roboflow YOLOv8 export directory")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu for CPU (default: auto)")
    args = parser.parse_args()

    runs_dir = os.path.join(HERE, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    data_yaml = write_dataset_yaml(args.data, os.path.join(runs_dir, "task1_data.yaml"))
    print(f"Dataset config written to {data_yaml}")

    # Start from COCO-pretrained YOLOv8n-pose and fine-tune on our devices.
    model = YOLO("yolov8n-pose.pt")
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=50,          # stop early if validation mAP stops improving
        cache=True,           # 60 images: keep the resized copies in RAM
        workers=4,
        seed=0,
        # Mixed precision gives NaN losses on GTX 16xx cards (no proper FP16
        # support), even though Ultralytics' AMP check passes. Train in FP32.
        amp=False,
        # Augmentation. The test set has rotated (~10-15 deg) and perspective
        # views, so add mild rotation/perspective on top of the defaults.
        degrees=15.0,
        perspective=0.0005,
        fliplr=0.5,           # safe now that flip_idx is correct
        project=runs_dir,
        name="task1_pose",
        exist_ok=True,
    )

    best = os.path.join(runs_dir, "task1_pose", "weights", "best.pt")
    os.makedirs(os.path.dirname(WEIGHTS_OUT), exist_ok=True)
    shutil.copy(best, WEIGHTS_OUT)
    print(f"Best weights copied to {WEIGHTS_OUT}")


if __name__ == "__main__":
    main()
