# Contributing

Thanks for your interest in this project! It began as my solution to the COMP3007 Machine Perception assignment at Curtin University (Semester 2, 2026). It's now a finished portfolio piece, but bug reports, questions and improvements are welcome.

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Academic integrity

**If you are a student taking COMP3007, or a similar unit, do not copy this code into your own assignment.** Submitting someone else's work is plagiarism under Curtin's academic integrity policy, whatever this repository's license says. You're welcome to read it to understand the techniques, then write your own solution.

For the same reason, pull requests that adapt the pipeline to a current semester's assignment spec won't be accepted.

## Ways to contribute

- **Report a bug.** Open an issue that includes:
  - the command you ran, e.g. `python assignment.py task4 path/to/photos`
  - the full terminal output
  - your OS, Python version and the versions of `torch`, `ultralytics` and `opencv-python`
  - if you can, an example image that fails. Only share photos you have the right to share
- **Suggest an improvement.** Open an issue first, so we can agree on the approach before you put time into a pull request.
- **Send a pull request** for bug fixes, robustness improvements (e.g. new lighting conditions or device models), documentation, or tooling.

## Development setup

```bash
git clone https://github.com/AstroPTP/medical-device-perception.git
cd medical-device-perception
pip install torch ultralytics opencv-python numpy scikit-learn matplotlib pillow
```

The project targets **Python 3.10**. The full Conda environment used during development is in `comp3007.yml`. Pre-trained weights in `data/` mean you can run inference straight away. See the [README](README.md#retrain-the-models) for retraining with your own dataset.

## Making changes

### Branches

Create a branch from `master` named `<type>/<short-description>`, for example:

```
feat/t1-device-detection
fix/thermo-tick-spacing
docs/training-guide
```

### Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/), in lowercase with a short imperative subject:

```
feat: add digit cnn and thermometer reading for task 3
fix: widen LCD edge search to ±35° so strong perspective views snap to the real border
docs: add pipeline figures and training visuals to README
chore: rename variables
```

Use `feat` for new behaviour, `fix` for bug fixes, `docs` for documentation only, and `chore` for comments, renames and housekeeping.

### Code style

Match the style of the existing files:

- **Docstrings** on every function, in this format:

  ```python
  def find_liquid_top(response, line, y_range=None):
      """
      find_liquid_top: One-line summary of what the function does

      :param response (np.ndarray): what it is
      :param line (tuple): what it is
      :return: what comes back (type)
      """
  ```

- **Comments explain why**, not what. For example, a comment on a threshold should say what it protects against.
- **No magic numbers in the code body.** Put tunable values in named module-level constants with a comment, as in `task1.py` (`DEFAULT_CONF`, `WORK_SIZE`, ...).
- **Resolution independence.** In Tasks 1 and 2, sizes and thresholds are fractions of the image size, not fixed pixel counts. Keep it that way.
- **Paths** are built from `HERE` (the script's folder), so scripts run from any working directory. Never hard-code a dataset location. Add a command-line option with a default inside the project folder.
- Keep the Apache 2.0 header at the top of each source file.

### Constraints to keep

These come from the original assignment and are part of what the project demonstrates:

- **No pre-made OCR** (Tesseract, EasyOCR, PaddleOCR, ...). Digit recognition must stay with the trained `DigitCNN`.
- `assignment.py` is the unit's framework and must stay unmodified. Put new behaviour in `task1.py` to `task4.py` and pass settings through `config.txt`.
- Output file names and locations (`output/taskN/...`) follow the assignment's spec, so don't change them.
- Inference should run on a CPU without user input or interactive plots.

### Testing your change

There is no automated test suite. Before you open a pull request:

1. Run the full pipeline on a folder of photos and check every output:

   ```bash
   python assignment.py task4 path/to/photos
   ```

2. If you changed one stage, also run that task on its own (`task1` to `task3`) and look at the images or text it writes to `output/`.
3. Make sure a photo with **no device** still produces **no output file**.
4. If you retrained a model, include the before and after metrics (YOLO `results.csv` or the confusion matrices in `runs/task3/`) in the pull request description.

Don't commit `output/`, `runs/`, datasets or downloaded base weights (`yolov8n-pose.pt`). Commit new weights in `data/` only if they replace the shipped models and you've included metrics.

## Pull request checklist

- [ ] The branch is up to date with `master`
- [ ] Commits follow the convention above
- [ ] New and changed functions have docstrings
- [ ] `python assignment.py task4 ...` runs end to end without errors
- [ ] The README is updated if behaviour, options or results changed

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE), the same as the rest of the project.
