## Summary

<!-- What does this PR change, and why? One or two sentences. -->

Closes #<!-- issue number, if any -->

## Type of change

- [ ] `feat`: new behaviour
- [ ] `fix`: bug fix
- [ ] `docs`: documentation only
- [ ] `chore`: comments, renames, housekeeping

## Affected stage(s)

- [ ] Task 1: detection / rectification
- [ ] Task 2: segmentation
- [ ] Task 3: digit recognition / thermometer reading
- [ ] Task 4: full pipeline
- [ ] Training scripts
- [ ] None (docs / repo only)

## How was this tested?

<!-- Commands you ran and what you checked. There is no automated test suite. -->

```bash
python assignment.py task4 path/to/photos
```

| Photo | Before | After |
|---|---|---|
| <!-- img4 --> | <!-- bpm 134,88,7 --> | <!-- bpm 134,88,75 --> |

- [ ] A photo with **no device** still produces **no output file**

## Model changes (if any)

<!-- Delete this section if no weights changed. -->

- [ ] Retrained `data/task1_pose.pt`: before/after metrics from `results.csv` are included below
- [ ] Retrained `data/task3_digits.pt`: before/after accuracy and confusion matrices are included below

## Screenshots

<!-- Before/after images for visual changes (rectification, crops, figures). Only images you have the right to share. -->

## Checklist

- [ ] The branch is up to date with `master`
- [ ] Commits follow the [commit convention](https://github.com/AstroPTP/medical-device-perception/blob/master/CONTRIBUTING.md#commit-messages)
- [ ] New and changed functions have docstrings in the project's format
- [ ] No hard-coded dataset paths; no `output/`, `runs/`, datasets or `yolov8n-pose.pt` committed
- [ ] `assignment.py` is unchanged, and no pre-made OCR was added
- [ ] The README is updated if behaviour, options or results changed
