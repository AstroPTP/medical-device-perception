
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
Train the Task 1 device detector: YOLOv8n-pose fine-tuned on the Roboflow
export "medical-device-perception" (YOLOv8 pose format)

Each label is one device box (class bpm or thermo) plus four keypoints, the
corners of the reading region in the device's own reading orientation:
0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left.
  - bpm    : corners of the LCD panel
  - thermo : corners of the printed scale area (top = the 50 degC end)

The checkpoint is copied to data/task1_pose.pt, which loads task1.py
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

# Absolute path of the directory containing this script, so paths resolve from any working directory
HERE = os.path.dirname(os.path.abspath(__file__))
# Default Roboflow YOLOv8 pose export, used when --data is not given
DEFAULT_DATA = os.path.join(
    HERE,
    "..",                                     # parent of the assignment folder
    "dataset",                                # shared dataset folder
    "medical-device-perception.v5i.yolov8",   # Roboflow export, version 5
)
# Destination of the best checkpoint, loaded by task1.py at inference time
WEIGHTS_OUT = os.path.join(HERE, "data", "task1_pose.pt")

# Keypoint tolerance (OKS sigma). Ultralytics hard-codes sigma = 1 / num_keypoints
# (0.25 here) for any non-human skeleton. That is loose, so predicted corners are
# only approximate (task1.py refines them classically). A tight value (0.05) was
# tried and made things worse: the OKS loss saturates for corners that start far
# away, so they get almost no gradient, and with ~34 thermometers whose scale
# corners are not sharp points, those corners collapsed. So the default keeps
# Ultralytics' value; --kpt-sigma remains for experiments.
DEFAULT_KPT_SIGMA = None

def set_keypoint_sigma(sigma):
    """
    set_keypoint_sigma: Patch Ultralytics so every keypoint uses the same OKS sigma
    in both the training loss and the validation metric

    :param sigma (float): OKS tolerance applied to all keypoints
    :return: None
    """
    original_loss_init = yolo_loss.v8PoseLoss.__init__

    def loss_init(self, model):
        """
        loss_init: Run the original v8PoseLoss constructor, then replace its keypoint
        loss with one that uses a uniform sigma

        :param self: v8PoseLoss instance being initialised
        :param model (torch.nn.Module): pose model the loss is attached to
        :return: None
        """
        original_loss_init(self, model)
        num_keypoints = self.kpt_shape[0]
        self.keypoint_loss = yolo_loss.KeypointLoss(
            sigmas=torch.full((num_keypoints,), sigma, device=self.device))

    original_init_metrics = PoseValidator.init_metrics

    def init_metrics(self, model):
        """
        init_metrics: Run the original PoseValidator metric setup, then override the
        per-keypoint sigma used for OKS

        :param self: PoseValidator instance
        :param model (torch.nn.Module): model being validated
        :return: None
        """
        original_init_metrics(self, model)
        self.sigma = np.full(self.kpt_shape[0], sigma)

    # Replace the Ultralytics methods with the wrappers above; every pose loss and
    # validator created after this point uses the uniform sigma.
    yolo_loss.v8PoseLoss.__init__ = loss_init
    PoseValidator.init_metrics = init_metrics

def write_dataset_yaml(dataset_dir, out_path):
    """
    write_dataset_yaml: Write a corrected copy of the Roboflow data.yaml

    The export sets flip_idx to [0, 1, 2, 3], which is wrong: a horizontal
    flip turns the top-left corner into the top-right one (and bottom-right
    into bottom-left), so the keypoint indices must be swapped as [1, 0, 3, 2].
    Without this, flip augmentation teaches the model mirrored corner orders
    Absolute paths are also used so training works from any directory

    :param dataset_dir (str): path to the Roboflow YOLOv8 export directory
    :param out_path (str): destination path for the corrected YAML file
    :return: path of the written YAML file (str)
    """
    dataset_dir = os.path.abspath(dataset_dir)
    with open(os.path.join(dataset_dir, "data.yaml")) as file:
        source = yaml.safe_load(file)

    config = {
        "path": dataset_dir,          # absolute dataset root; split folders below are relative to it
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "kpt_shape": [4, 3],          # 4 corner keypoints, each stored as (x, y, visibility)
        "flip_idx": [1, 0, 3, 2],     # corrected horizontal-flip mapping: swap left and right corners
        "nc": source["nc"],           # class count kept from the Roboflow export
        "names": source["names"],     # class names (bpm, thermo) kept from the Roboflow export
    }
    with open(out_path, "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return out_path

def main():
    """
    main: Parse command-line arguments, write the corrected dataset config,
    fine-tune YOLOv8n-pose and copy the best checkpoint to data/task1_pose.pt

    :return: None
    """
    # Command-line options; every option has a default, so the script runs with no arguments.
    parser = argparse.ArgumentParser(description="Train the Task 1 YOLOv8n-pose detector")
    # Dataset location (Roboflow YOLOv8 pose export).
    parser.add_argument("--data", default=DEFAULT_DATA, help="Roboflow YOLOv8 export directory")
    # Maximum number of training epochs; early stopping may end training sooner.
    parser.add_argument("--epochs", type=int, default=150)
    # Training image size in pixels; 960 keeps small LCD digits and scale marks visible.
    parser.add_argument("--imgsz", type=int, default=960)
    # Images per batch; kept small to fit in GPU memory at 960 px.
    parser.add_argument("--batch", type=int, default=8)
    # Training device; None lets Ultralytics pick the GPU if one is available.
    parser.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu for CPU (default: auto)")
    parser.add_argument(
        "--kpt-sigma",
        type=float,
        default=DEFAULT_KPT_SIGMA,   # None keeps Ultralytics' built-in sigma
        help="keypoint OKS sigma for loss and validation "
             "(default: Ultralytics' 1/num_keypoints = 0.25)",
    )
    args = parser.parse_args()

    # Override the keypoint sigma only when --kpt-sigma is given; otherwise keep Ultralytics' default
    if args.kpt_sigma is not None:
        set_keypoint_sigma(args.kpt_sigma)
        print(f"Keypoint sigma set to {args.kpt_sigma}")

    # Output folder for the dataset config and the training run
    runs_dir = os.path.join(HERE, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    # Write the corrected data.yaml (fixed flip_idx, absolute paths) and train from it instead of the raw export
    data_yaml = write_dataset_yaml(args.data, os.path.join(runs_dir, "task1_data.yaml"))
    print(f"Dataset config written to {data_yaml}")

    # Start from COCO-pretrained YOLOv8n-pose and fine-tune on the device dataset.
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
        amp=False,            # FP16 gives NaN losses on GTX 16xx cards, so train in FP32
        degrees=15.0,         # mild rotation: the test set has ~10-15 deg rotated views
        perspective=0.0005,   # mild perspective warp for angled test views
        fliplr=0.5,           # safe now that flip_idx is correct
        project=runs_dir,
        name="task1_pose",
        exist_ok=True,
    )

    # Checkpoint with the best validation fitness, saved by Ultralytics during training.
    best = os.path.join(
        runs_dir,
        "task1_pose",   # run name passed to model.train
        "weights",      # Ultralytics checkpoint folder
        "best.pt",      # best epoch, as opposed to last.pt
    )
    # Copy the best checkpoint to the fixed path that task1.py loads.
    os.makedirs(os.path.dirname(WEIGHTS_OUT), exist_ok=True)
    shutil.copy(best, WEIGHTS_OUT)
    print(f"Best weights copied to {WEIGHTS_OUT}")

if __name__ == "__main__":
    main()
