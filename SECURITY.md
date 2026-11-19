# Security Policy

## Project scope

This is a finished university project (COMP3007 Machine Perception, Curtin University, 2026), maintained on a best-effort basis by one person. It is a command-line research and portfolio tool, **not** a medical device.

> **Not for clinical use.** The readings it produces have not been clinically validated and must not be used for diagnosis, treatment or any other health decision. An incorrect reading is a known limitation, not a security vulnerability. Please report it as a [bug](https://github.com/AstroPTP/medical-device-perception/issues/new/choose) instead.

## Supported versions

Only the latest commit on `master` is supported. There are no releases or backported fixes.

| Version | Supported |
|---|---|
| `master` (latest) | ✅ |
| Older commits | ❌ |

## Reporting a vulnerability

**Please don't open a public issue for security problems.**

Report them privately through GitHub instead:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability**.
3. Include:
   - what the problem is and which file(s) it affects
   - steps or a minimal example to reproduce it
   - the impact you think it has
   - your environment (OS, Python, `torch` and `ultralytics` versions)

**What to expect:**

- An acknowledgement within **7 days**
- An assessment, and a fix or explanation, within **30 days** where possible
- Credit in the fix commit if you'd like it (say so in your report)

As a solo, unpaid project, there is no bug bounty.

## Known risks and safe use

### Model files can run code

PyTorch `.pt` checkpoints are Python pickle files. **Loading one from an untrusted source can execute arbitrary code on your machine.**

- `task1.py` loads `data/task1_pose.pt` through Ultralytics, which unpickles the full model.
- `task3.py` loads `data/task3_digits.pt` with `torch.load`, which is restricted to plain weights (`weights_only=True`) by default in PyTorch 2.6 and later. Older versions don't have this protection.

**Only use the weights shipped in this repository, or ones you trained yourself.** Treat a pull request that replaces files in `data/` with the same care as one that changes code.

### Input images

The pipeline decodes images with OpenCV. Image decoders have had memory-safety bugs in the past, so keep `opencv-python` up to date if you process images from sources you don't trust.

### Dependencies

Vulnerabilities in PyTorch, Ultralytics, OpenCV or other dependencies should be reported to those projects directly. If one affects how this project uses the library, you're welcome to report it here too.

## Out of scope

- Wrong or missing readings, misdetections and other accuracy problems (report these as bugs)
- Problems that need an attacker who can already modify files in your copy of the repository
- Vulnerabilities in third-party dependencies that this project doesn't make worse
