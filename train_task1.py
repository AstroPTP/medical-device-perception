
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
Train the Task 1 device detector: YOLOv8n-pose fine-tuned on the Roboflow
export "medical-device-perception" (YOLOv8 pose format).

Each label is one device box (class bpm or thermo) plus four keypoints, the
corners of the reading region in the device's own reading orientation:
0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left.
  - bpm    : corners of the LCD panel
  - thermo : corners of the printed scale area (top = the 50 degC end)

The best checkpoint is copied to data/task1_pose.pt, which task1.py loads.

Usage (inside the comp3007 conda environment):
    python train_task1.py
    python train_task1.py --data ../dataset/medical-device-perception.v5i.yolov8 --epochs 150
"""

import argparse
import os
import shutil

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.utils import loss as yolo_loss

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "dataset", "medical-device-perception.v5i.yolov8")
WEIGHTS_OUT = os.path.join(HERE, "data", "task1_pose.pt")

# Keypoint tolerance (OKS sigma) for our 4 corners. Ultralytics hard-codes
# sigma = 1 / num_keypoints (0.25 here) for any non-human skeleton, 3-10x
# looser than COCO's human keypoints (0.025-0.107). With 0.25 a corner that is
# 100+ px off barely raises the loss, and validation still counts it as
# correct, so the model never learns precise corners and best.pt is chosen on
# a misleading score. Corners of a printed scale / LCD are sharp, well-defined
# points, so a COCO-like value is appropriate.
DEFAULT_KPT_SIGMA = 0.05


def set_keypoint_sigma(sigma):
    """Use `sigma` for every keypoint in both the training loss and validation OKS."""
    original_loss_init = yolo_loss.v8PoseLoss.__init__

    def loss_init(self, model):
        original_loss_init(self, model)
        nkpt = self.kpt_shape[0]
        self.keypoint_loss = yolo_loss.KeypointLoss(
            sigmas=torch.full((nkpt,), sigma, device=self.device))

    original_init_metrics = PoseValidator.init_metrics

    def init_metrics(self, model):
        original_init_metrics(self, model)
        self.sigma = np.full(self.kpt_shape[0], sigma)

    yolo_loss.v8PoseLoss.__init__ = loss_init
    PoseValidator.init_metrics = init_metrics


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
    parser.add_argument("--kpt-sigma", type=float, default=DEFAULT_KPT_SIGMA,
                        help="keypoint OKS sigma for loss and validation (Ultralytics default: 0.25)")
    args = parser.parse_args()

    set_keypoint_sigma(args.kpt_sigma)
    print(f"Keypoint sigma set to {args.kpt_sigma}")

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
