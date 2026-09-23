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

This version intentionally stops at row tracking. Hero recognition, username OCR,
team-color classification, and event-icon classification are separate modules to
add after row detection is stable.


## Real-time viewing and audio

The application now opens two debugging windows by default:

- **Overwatch Vision - Full View**: the full game client with the normalized
  kill-feed search region and detected kill-feed rows boxed.
- **Overwatch Vision - Kill Feed Monitor**: an enlarged live crop of the
  top-right kill-feed area. Green boxes show current row candidates and white
  boxes/IDs show confirmed temporal tracks.

When a newly confirmed kill-feed row appears after the startup baseline period,
the app:
1. prints the event to the terminal,
2. shows an `Elimination detected` banner in the debug views,
3. says **"Elimination detected"** using local text-to-speech.

Audio is intentionally generic in this milestone because hero/name parsing has
not been implemented yet. Later, the announcer can receive parser output such as
`Tracer eliminated Ana`.

Audio settings and viewing-window scales are in `config/settings.yaml`.


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

Later runs reuse the existing virtual environment and launch the project directly.
