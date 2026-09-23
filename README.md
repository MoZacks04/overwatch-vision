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
