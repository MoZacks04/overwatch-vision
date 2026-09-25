# Overwatch Vision

A scalable, passive computer-vision project for reading Overwatch HUD state from the screen.

The first milestone is real-time kill-feed row tracking:
- finds the Overwatch client area on Windows
- uses normalized HUD regions rather than fixed pixels
- detects variable-width kill-feed rows
- tracks rows across frames even when they move vertically
- emits a `new_row` event when a genuinely new row appears
- provides a debug preview

This project does not control mouse or keyboard input.

## Project structure

```text
overwatch_vision_project/
├─ run.py
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ config/
│  └─ settings.yaml
├─ src/
│  └─ overwatch_vision/
│     ├─ __init__.py
│     ├─ app.py
│     ├─ capture.py
│     ├─ models.py
│     ├─ regions.py
│     ├─ debug_view.py
│     ├─ killfeed/
│     │  ├─ __init__.py
│     │  ├─ detector.py
│     │  ├─ row_detector.py
│     │  ├─ row_normalizer.py
│     │  ├─ tracker.py
│     │  └─ parser.py
│     └─ utils/
│        ├─ __init__.py
│        ├─ geometry.py
│        └─ image_ops.py
└─ tests/
   ├─ __init__.py
   └─ test_geometry.py
```

## Setup

From PowerShell:

```powershell
cd path\to\overwatch_vision_project

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Open Overwatch first, then:

```powershell
python .\run.py
```

Controls:
- `Q` = quit
- `D` = toggle debug overlays
- `R` = reset the kill-feed tracker

## Design

1. `capture.py` finds the Overwatch client area.
2. `regions.py` extracts a normalized top-right search region.
3. `row_detector.py` finds saturated HUD components and groups them by vertical center.
4. Each row width is determined dynamically, so username length can vary.
5. `row_normalizer.py` preserves aspect ratio and right-aligns each row in a fixed canvas.
6. `tracker.py` matches rows across frames even when their Y position changes.
7. A new row is emitted only after it persists for several frames.

## Current milestone

The project now goes beyond row tracking. For each confirmed elimination row it
attempts to recover:

- attacker and victim team color,
- attacker and victim player names with OCR,
- attacker and victim hero identity,
- friendly/enemy alive counts from the small status widget above the feed.

Hero recognition uses a locally cached reference set. Low-confidence rows are
saved under `debug_frames/killfeed_review` so the recognizer can be tuned from
real examples without recording the whole screen.

Ability icons, headshot/critical markers, assists, resurrection rows, and other
special kill-feed variants are represented in the event model but are not yet
fully classified.


## Real-time viewing and audio

The application now opens two debugging windows by default:

- **Overwatch Vision - Full View**: the full game client with the normalized
  kill-feed search region and detected kill-feed rows boxed.
- **Overwatch Vision - Kill Feed Monitor**: an enlarged live crop of the
  top-right kill-feed area. Green boxes show current row candidates and white
  boxes/IDs show confirmed temporal tracks.

When a newly confirmed elimination appears, the app prints the parsed details,
shows the event in the debug view, and speaks a contextual phrase such as
`Enemy Doomfist eliminated your Mercy` when hero/team parsing succeeds.

The app also speaks **"Overwatch Vision audio ready"** on startup so audio can
be tested immediately. Windows SAPI is the preferred speech backend, with
pyttsx3 as a fallback.

On the first run, OCR may download its recognition model and hero references may
be downloaded into `.cache/hero_portraits`. Audio, OCR, team status, and hero
recognition settings are all in `config/settings.yaml`.


## Easiest Windows launch

On Windows, you can now use the included launcher instead of manually creating
and activating a virtual environment.

After pulling the latest repository, either double-click:

```text
start_windows.bat
```

or run it from the VS Code terminal:

```powershell
.\start_windows.bat
```

On the first run it automatically:
1. checks that Python is installed,
2. creates `.venv`,
3. installs the packages from `requirements.txt`,
4. starts `run.py`.

Later runs reuse the existing virtual environment. The launcher still checks
`requirements.txt` each time so newly added dependencies are installed after a
`git pull`.


## Performance scheduling

The real-time loop is now separated from the expensive recognition work:

- screen capture and the preview target 30 FPS,
- kill-feed row detection runs every 3 captured frames (about 10 Hz),
- team-status analysis runs every 15 captured frames (about 2 Hz),
- the team counter uses lightweight digit templates instead of EasyOCR,
- username OCR and hero parsing run on a background queue only when a new
  elimination row is confirmed.

This means a slow OCR call should no longer freeze screen capture or prevent a
short-lived elimination row from being recorded. The debug windows show the
current parse-queue length so backlog is visible while testing.

These rates can be changed under `performance:` in
`config/settings.yaml`.


## Kill-feed V3 training workflow

The project now supports a staged path away from generic color/portrait guessing:

1. The live detector searches only a tight normalized top-right kill-feed ROI.
2. Each confirmed row produces exact killer/victim portrait crops.
3. `label_hero_samples.py` turns those real in-game crops into labeled hero
   examples under `.cache/killfeed_hero_templates/`.
4. `train_hero_classifier.py` trains a tiny 64x64 CNN on those exact kill-feed
   portraits and exports a TorchScript model under `models/`.
5. At runtime, the trained classifier is preferred. If it is uncertain or no
   model exists yet, the conservative template/generic recognizer remains as a
   fallback.
6. Hero identity still requires multi-frame consensus before a spoken call, and
   parsed events still pass through the 10-second semantic duplicate guard.

### Labeling and training the row verifier

Existing low-confidence row crops under `debug_frames/killfeed_review/` can be
turned into a binary dataset without recording more games first.

Run:

```powershell
.\\.venv\\Scripts\\python.exe .\\label_killfeed_rows.py
```

Controls:
- `R` = real Overwatch kill-feed row
- `F` = false detection / scenery / malformed crop
- `S` = skip if unsure
- `Q` or `Esc` = quit

Progress is saved after every image. Labeled copies are stored locally under:

```text
.cache/killfeed_row_verifier/real/
.cache/killfeed_row_verifier/false/
```

After both classes have useful examples, train:

```powershell
.\\.venv\\Scripts\\python.exe .\\train_killfeed_row_verifier.py
```

The trainer keeps captures from the same short time window together during the
train/validation split so near-duplicate frames are less likely to inflate the
held-out score. It exports:

```text
models/killfeed_row_verifier.pt
models/killfeed_row_verifier_labels.json
```

On restart, the current red/blue geometry detector still proposes candidate
rows, but the learned verifier rejects proposals that do not visually resemble
a real kill-feed entry. If no verifier model exists, runtime behavior is
unchanged. This verifier reduces false positives; a later full-ROI localization
model will be needed to recover rows that the color detector never proposes.

### Collecting row examples

Run:

```powershell
.\.venv\Scripts\python.exe .\collect_killfeed_training_data.py
```

Controls:
- `S` saves the current tight kill-feed ROI.
- `A` toggles auto-save while row candidates are visible.
- `Q` quits.

The collector stores the raw ROI plus current weak row labels under
`datasets/killfeed_rows/raw/`. These examples can later be manually reviewed
and used to train a one-class row detector if geometry-based detection still
produces false positives.

### Training hero identity

After normal play has produced samples under `debug_frames/hero_samples/`:

```powershell
.\.venv\Scripts\python.exe .\label_hero_samples.py
```

Label clean hero portraits, then train:

```powershell
.\.venv\Scripts\python.exe .\train_hero_classifier.py
```

Restart Overwatch Vision after training. If the model files exist, the runtime
loads them automatically.
