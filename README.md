# SafeVision — Face Mask Compliance Detection System

A command-line Computer Vision system that detects faces in images or video,
classifies each face as **wearing a mask** or **not wearing a mask** using a
Convolutional Neural Network, stores every detection in a database, and
generates compliance analytics reports.

Built for the **Computer Vision** course project. Everything runs from a
terminal — no GUI required.

---

## Table of Contents

1. [What the system does](#1-what-the-system-does)
2. [Requirements](#2-requirements)
3. [Setup — step by step](#3-setup--step-by-step)
4. [Quick start (5 commands)](#4-quick-start-5-commands)
5. [Full command reference](#5-full-command-reference)
6. [Project structure](#6-project-structure)
7. [Architecture & workflow](#7-architecture--workflow)
8. [Functional requirements](#8-functional-requirements)
9. [Non-functional requirements](#9-non-functional-requirements)
10. [Using a real dataset](#10-using-a-real-dataset)
11. [Running the tests](#11-running-the-tests)
12. [Troubleshooting](#12-troubleshooting)
13. [Known limitations](#13-known-limitations)

---

## 1. What the system does

A workplace, campus or clinic needs to know whether people entering a space are
wearing face masks. Manually watching a camera feed does not scale. SafeVision
automates the check:

```
image / video frame
        ↓
 face detection  (OpenCV Haar cascade)
        ↓
 per-face crop → normalize → CNN classifier
        ↓
 with_mask / without_mask  +  confidence
        ↓
 persisted to SQLite  →  compliance report (CSV + summary)
```

Every detection is attributed to the logged-in user and timestamped, so the
record set is auditable and a human reviewer can correct a misclassification.

---

## 2. Requirements

- **Python 3.9 – 3.12**
- **pip**
- ~1.5 GB disk space (TensorFlow is the bulk of it)
- No GPU required — the CNN is small and trains on CPU in under a minute
- No internet connection required after installing dependencies

Verify your Python version:

```bash
python --version
```

> On some systems the command is `python3` / `pip3`. Substitute accordingly
> throughout this README.

---

## 3. Setup — step by step

### Step 1 — Clone the repository

```bash
git clone https://github.com/{github-username}/{repo-name}.git
cd {repo-name}
```

### Step 2 — Create a virtual environment

This keeps the project's dependencies isolated from your system Python.

**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (cmd)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

Your prompt should now be prefixed with `(.venv)`.

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs `opencv-python`, `numpy` and `tensorflow`. Installation takes a
few minutes, mostly for TensorFlow.

> **Lower-memory machines:** `tensorflow-cpu` is a smaller download and works
> identically here. Install it with
> `pip install opencv-python numpy tensorflow-cpu` instead.

### Step 4 — Verify the installation

```bash
python -c "import cv2, numpy, tensorflow; print('All dependencies OK')"
```

### Step 5 — Initialize the system

```bash
python main.py init
```

This creates the SQLite database, the default admin account, and six synthetic
sample images you can immediately run detection on. There is **no
configuration file to edit** — all paths and thresholds live in `config.py`
and have working defaults.

**Default credentials:** username `admin`, password `admin123`

---

## 4. Quick start (5 commands)

Run these in order from the project root, after completing Setup above.

```bash
# 1. Initialize database, admin account and sample images
python main.py init

# 2. Train the CNN classifier (~1 minute on CPU)
python train_model.py --epochs 15

# 3. Run detection on the bundled sample images
python main.py detect-image --path data/sample_images --save-annotated -u admin -p admin123

# 4. View the stored detection records
python main.py records -u admin -p admin123

# 5. Generate a compliance report
python main.py report -u admin -p admin123
```

To avoid typing credentials each time, export them once:

```bash
export SAFEVISION_USER=admin
export SAFEVISION_PASS=admin123     # Windows: set SAFEVISION_USER=admin
```

Then every command can be run without `-u` / `-p`.

---

## 5. Full command reference

Every command supports `--help`:

```bash
python main.py --help
python main.py detect-image --help
```

### `init` — first-time setup
```bash
python main.py init
```
Creates database tables, the default admin user, and sample images.

### `register` — add a user
```bash
python main.py register alice --new-password secret123 --role operator
python main.py register bob   --new-password secret123 --role admin
```
Roles are `operator` (can detect, view, correct records) and `admin`
(everything, plus deleting records). Omit `--new-password` to be prompted
securely instead of putting the password in your shell history.

### `detect-image` — analyze an image or a folder
```bash
# Single image
python main.py detect-image --path /path/to/photo.jpg

# Whole folder
python main.py detect-image --path data/sample_images

# Also write annotated copies with boxes drawn on them
python main.py detect-image --path data/sample_images --save-annotated
```
Annotated images are written to `reports/annotated/`. Green box = mask
detected, red box = no mask.

### `detect-video` — analyze a video file or webcam
```bash
# Video file, analyzing every 15th frame
python main.py detect-video --source clip.mp4 --every-n 15

# Webcam (index 0), stop after 20 analyzed frames
python main.py detect-video --source 0 --max-frames 20

# Write an annotated output video
python main.py detect-video --source clip.mp4 --output reports/annotated.mp4
```
Add `--display` to open a live preview window — this requires a desktop
environment and is **off by default** so the tool works over SSH.

### `records` — CRUD on stored detections
```bash
python main.py records --limit 20                      # list (Read)
python main.py records --filter without_mask           # list violations only
python main.py records --update 3 --label with_mask    # correct a label (Update)
python main.py records --delete 3                      # delete (admin only)
```
Records are **created** automatically by the `detect-*` commands.

### `evaluate` — model quality metrics
```bash
python main.py evaluate --samples-per-class 100
python main.py evaluate --data-dir data/test
```
Prints accuracy, per-class precision / recall / F1, and a confusion matrix.

### `report` — compliance analytics
```bash
python main.py report
```
Prints a summary and writes a timestamped `.csv` + `.txt` report into
`reports/`.

### `train_model.py` — train the classifier
```bash
python train_model.py                                  # synthetic data
python train_model.py --epochs 20 --batch-size 32
python train_model.py --data-dir data/train --epochs 15  # real dataset
```

---

## 6. Project structure

```
SafeVision/
├── main.py                  # CLI entry point — all user interaction
├── train_model.py           # Training pipeline script
├── config.py                # Central configuration (paths, thresholds)
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── auth.py              # Module 1: user management, RBAC, password hashing
│   ├── preprocessing.py     # Module 2: image loading, resizing, cropping, augmentation
│   ├── detection.py         # Module 3: face detection + mask classification pipeline
│   ├── records.py           # Module 4: CRUD over detection records
│   ├── analytics.py         # Module 5: compliance statistics and report generation
│   ├── model.py             # CNN architecture, train / save / load
│   ├── evaluation.py        # Accuracy, precision, recall, F1, confusion matrix
│   ├── video_stream.py      # Video-file and webcam handling
│   ├── utils.py             # Shared helpers + synthetic dataset generator
│   └── logger_setup.py      # Rotating-file logging configuration
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_detection.py
│   └── test_auth_records.py
│
├── data/sample_images/      # Generated demo images
├── models/                  # Trained model artifacts (git-ignored)
├── database/                # SQLite database (git-ignored)
├── logs/                    # Rotating application logs (git-ignored)
└── reports/                 # Generated CSV / text reports (git-ignored)
```

---

## 7. Architecture & workflow

The system uses a **layered architecture**. Each layer only talks to the layer
below it, which keeps modules independently testable.

```
┌──────────────────────────────────────────────┐
│  Presentation layer      main.py (argparse)  │
├──────────────────────────────────────────────┤
│  Application layer                           │
│    auth.py    records.py    analytics.py     │
├──────────────────────────────────────────────┤
│  Domain / CV layer                           │
│    detection.py   model.py   evaluation.py   │
│    preprocessing.py   video_stream.py        │
├──────────────────────────────────────────────┤
│  Infrastructure                              │
│    SQLite   file system   logger_setup.py    │
└──────────────────────────────────────────────┘
```

**Typical user workflow**

1. User runs `init` once → database and admin account exist.
2. User trains the model with `train_model.py` → `models/mask_classifier.keras`.
3. User authenticates on every command via `-u/-p`, env vars, or a prompt.
4. User runs `detect-image` or `detect-video`:
   - input is decoded and validated (`preprocessing.py`)
   - faces are located (`detection.py` → Haar cascade)
   - each face crop is normalized and classified (`model.py`)
   - results are written to the database (`records.py`)
5. User reviews results with `records`, correcting any mistakes with `--update`.
6. User generates a compliance report with `report` (`analytics.py`).

**Input / output contract**

| Stage | Input | Output |
|---|---|---|
| Preprocessing | file path or BGR frame | `float32` array, shape `(100, 100, 3)`, range `[0,1]` |
| Face detection | BGR image | list of `(x, y, w, h)` boxes |
| Classification | face crop | `(label, confidence)` where label ∈ {`with_mask`, `without_mask`} |
| Persistence | `Detection` + source + user | integer `record_id` |
| Analytics | database rows | summary dict + CSV/TXT files |

---

## 8. Functional requirements

The project implements **five** major functional modules (three were required).

| # | Module | File | Capability |
|---|---|---|---|
| 1 | **User management** | `src/auth.py` | Registration, login, PBKDF2 password hashing, `admin`/`operator` role-based access control |
| 2 | **Data input & processing** | `src/preprocessing.py`, `src/video_stream.py` | Load images, folders, video files and webcam streams; resize, normalize, crop, augment; frame skipping |
| 3 | **Prediction / classification** | `src/detection.py`, `src/model.py` | Haar-cascade face detection + CNN mask classification with confidence scores and annotated output |
| 4 | **CRUD operations** | `src/records.py` | Create (automatic on detection), Read (list/filter/get), Update (reviewer label correction), Delete (admin-only) |
| 5 | **Reporting & analytics** | `src/analytics.py`, `src/evaluation.py` | Compliance rate, per-source breakdown, CSV export; model accuracy / precision / recall / F1 / confusion matrix |

---

## 9. Non-functional requirements

Seven are implemented (four were required).

| # | Requirement | How it is met | Where |
|---|---|---|---|
| 1 | **Security** | Passwords hashed with PBKDF2-HMAC-SHA256, 100,000 iterations, unique 16-byte random salt per user; constant-time comparison via `hmac.compare_digest` (resists timing attacks); role-based authorization on destructive operations; parameterized SQL everywhere (no injection); passwords readable from env vars or a hidden prompt so they need not appear in shell history | `src/auth.py`, `src/records.py` |
| 2 | **Error handling** | Custom exception types (`AuthError`, `PreprocessingError`, `DetectionError`, `VideoError`); unreadable files are skipped rather than aborting a batch; a top-level handler in `main.py` converts any unexpected exception into a clean message plus a distinct exit code (`0` OK, `1` error, `2` auth failure) for scripting | `main.py` and all modules |
| 3 | **Logging & monitoring** | Single rotating-file logger (2 MB × 3 backups) capturing every detection, login attempt, CRUD operation and error, with timestamps and severity; console shows only warnings and above to keep CLI output readable | `src/logger_setup.py` |
| 4 | **Performance** | Compact CNN (~0.2M parameters) trains in under a minute on CPU; Haar cascade is far cheaper than a deep detector; `--every-n` frame skipping bounds video cost; the trained model and cascade are loaded once and cached in module-level globals rather than per image | `src/model.py`, `src/detection.py`, `src/video_stream.py` |
| 5 | **Reliability** | Fails safe rather than failing silently — with no trained model the system raises a typed error with an actionable instruction instead of emitting unreliable guesses; whole-image fallback when no face is found, explicitly flagged in output; `try/finally` guarantees video capture and writer handles are released even on exception or interrupt | `src/detection.py`, `src/video_stream.py` |
| 6 | **Maintainability** | Layered architecture with single-responsibility modules; all tunables centralized in `config.py`; type hints and docstrings throughout; 42 unit tests; consistent naming | whole project |
| 7 | **Usability & resource efficiency** | Discoverable `--help` on every subcommand; sensible defaults so `init` needs no arguments; headless by default (`--display` is opt-in) so it runs over SSH; TensorFlow imported lazily so lightweight commands like `records` start fast; SQLite needs no database server | `main.py`, `src/model.py` |

---

## 10. Using a real dataset

The default pipeline trains on **synthetic generated images** so the project is
runnable with zero downloads. For realistic accuracy, substitute a real
dataset — for example the widely used *Face Mask Detection* dataset on Kaggle.

Arrange the images in this structure:

```
data/train/
├── with_mask/
│   ├── img001.jpg
│   └── ...
└── without_mask/
    ├── img101.jpg
    └── ...
```

Then train and evaluate:

```bash
python train_model.py --data-dir data/train --epochs 15
python main.py evaluate --data-dir data/test
```

The class folder names must exactly match `CLASS_NAMES` in `config.py`.
No code changes are needed — only the folder of images.

---

## 11. Running the tests

```bash
python -m unittest discover -s tests -v
```

Expected: **42 tests, all passing.** The suite covers:

- **`test_preprocessing.py`** — resizing, normalization ranges, crop clipping at
  image boundaries, augmentation, and failure paths for missing/corrupt files
- **`test_detection.py`** — face-detection output format, classifier label
  validity and confidence bounds, and that annotation does not mutate the
  caller's image
- **`test_auth_records.py`** — password hashing (no plaintext storage, distinct
  hashes for identical passwords), rejection of wrong/duplicate/weak
  credentials, role enforcement, the full CRUD cycle, and compliance-rate maths

Tests run against a temporary SQLite database, so your real data is untouched.

---

## 12. Troubleshooting

**`ModuleNotFoundError: No module named 'cv2'` / `'tensorflow'`**
The virtual environment is not active or dependencies were not installed.
Re-run Step 2 and Step 3 of Setup.

**`[SETUP REQUIRED] No trained model is available...`**
Run `python train_model.py --epochs 15` first. Detection deliberately refuses to
classify without a trained model rather than emitting unreliable guesses.

**TensorFlow prints `oneDNN` / `AVX2` messages at startup**
These are informational, not errors. Silence them with:
```bash
export TF_CPP_MIN_LOG_LEVEL=2
```

**`Could not open video source: 0`**
No webcam is available (common on servers). Use a video file with
`--source path/to/clip.mp4` instead.

**`cv2.error` mentioning a display or GTK when using `--display`**
You are on a headless machine. Drop `--display` and use `--output` to write an
annotated video file instead.

**Detection reports `[whole-image]` instead of a face box**
No face was located, so the whole frame was classified as one region. This is
expected for the synthetic sample images and for tightly cropped face photos.
See [Known limitations](#13-known-limitations).

**Permission errors writing to `logs/` or `database/`**
Run from the project root so the relative paths in `config.py` resolve, and
ensure your user can write to the project directory.

---

## 13. Known limitations

Stated openly, since honest evaluation is part of the project:

1. **The synthetic dataset is trivially separable.** Held-out accuracy lands
   between 0.94 and 1.00 (measured across three runs at 15 epochs) because the
   generated "masked" images contain a flat coloured rectangle that the CNN
   learns almost immediately. **These numbers should not be read as real-world
   accuracy.** Use a real dataset ([Section 10](#10-using-a-real-dataset)) for a
   meaningful figure. Training for too few epochs (8) drops accuracy to ~0.875,
   so the `--epochs 15` default in the quick start is the recommended setting.

2. **Haar cascades only reliably detect frontal, unoccluded faces.** Profile
   views, strong backlighting and heavy occlusion cause misses. A mask itself
   occludes part of the face, so masked people are detected somewhat less
   reliably than unmasked ones. A modern detector (MTCNN, YOLO-face, or
   OpenCV's DNN face detector) would improve recall at a higher compute cost.

3. **Haar does not detect the bundled synthetic sample images at all**, because
   they are schematic drawings rather than photographs. The whole-image
   fallback covers this case and is explicitly flagged as `[whole-image]` in
   the output so it is never mistaken for a genuine face detection.

4. **Classification requires a trained model — the system will not guess.** An
   earlier version fell back to an edge-density heuristic when no model was
   present. Measurement showed the two classes' edge-density distributions
   overlapped completely, and it labelled all six sample images `with_mask`,
   including the three unmasked ones. Since a wrong `with_mask` is a violation
   nobody gets alerted to, that fallback was removed. `detect-image` and
   `detect-video` now exit with code 1 and an instruction to run
   `train_model.py` instead.

5. **No face recognition or identity tracking.** The system counts compliance
   events; it does not identify individuals or track a person across frames.
   This is a deliberate privacy-preserving scope decision.

6. **Single-machine scope.** SQLite and a local file system are appropriate for
   one camera and a coursework deployment. A multi-camera deployment would need
   a client/server database and a job queue.

---

## License

Submitted as academic coursework for the Computer Vision course.

## Author 

- Name : Manthan Awagan
- Reg. No. : 24BAI10381
- Course : Computer Vision
- Date : 16 Sept 2026
