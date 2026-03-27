# AutoLab — Closed-Loop Liquid Handling Optimiser

AutoLab is a self-driving laboratory system that autonomously optimises OT-2 pipette parameters using Bayesian optimisation. It runs experiments, captures images, analyses liquid detection accuracy, and feeds results back to the optimiser — without human intervention between iterations.

---

## How it works

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Closed-Loop Campaign                            │
│                                                                     │
│   ┌──────────────┐     suggest      ┌──────────────────────────┐   │
│   │  Bayesian    │ ─────────────── ▶│   Protocol Generator     │   │
│   │  Optimiser   │                  │   (per-iteration .py)    │   │
│   │  (Gaussian   │                  └────────────┬─────────────┘   │
│   │   Process)   │                               │ upload + run    │
│   │              │                  ┌────────────▼─────────────┐   │
│   │              │ ◀─── accuracy ── │   OT-2 Robot             │   │
│   └──────────────┘                  │   (HTTP REST API)        │   │
│                                     └────────────┬─────────────┘   │
│                                                  │ captures images  │
│                                     ┌────────────▼─────────────┐   │
│                                     │   Camera + Liquid        │   │
│                                     │   Analyser               │   │
│                                     │   (YOLO-NAS + HSV)       │   │
│                                     └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

Each iteration: the optimiser suggests parameters → a protocol file is generated and uploaded to the robot → the robot runs it while a camera captures tip images → the liquid analyser scores each image → the score is fed back to the optimiser. This repeats until convergence or the experiment cap is reached.

---

## Project Structure

```
autolabproject/
│
├── experiments/                     # Main campaign orchestration
│   ├── experiment_runner.py         # Entry point — runs the full campaign loop
│   ├── campaign_config.py           # User-editable settings (factors, limits, IPs)
│   ├── campaign_dashboard.py        # Live HTTP dashboard with pause/stop controls
│   ├── setup_wizard.py              # 5-step web wizard that runs before the campaign
│   ├── protocol_generator.py        # Generates a .py protocol file per iteration
│   ├── protocols/                   # Generated protocol files (git-ignored)
│   └── results/                     # Per-iteration results, images, analysis CSVs
│       ├── results.csv              # Full campaign log (all iterations, parameters, accuracy)
│       ├── campaign_log.csv         # Optimiser internal log
│       ├── convergence_plot.png     # Accuracy-vs-iteration plot saved at campaign end
│       └── iter_NNN/                # Per-iteration folder
│           ├── images/              # Captured tip images for that iteration
│           └── analysis/            # Liquid analysis output (CSV + annotated images)
│
├── OT2_operation/                   # Robot control
│   ├── ot2_controller.py            # OT-2 HTTP REST client (no Opentrons App needed)
│   └── run_protocol.py              # Universal protocol runner (any .py file)
│
├── camera/                          # Camera and image analysis
│   ├── camera_server.py             # HTTP server that captures images from USB camera
│   ├── camera_live_view.py          # Live preview utility for camera positioning
│   ├── analysis_script/
│   │   └── liquid_analysis.py       # YOLO-NAS + HSV liquid detection pipeline
│   └── protocol/                    # Example and test OT-2 protocols with camera capture
│
├── simulation/                      # Bayesian optimisation engine
│   ├── simulation.py                # DOEOptimiser class (Gaussian Process + LHS + EI)
│   └── dashboard.py                 # Static HTML dashboard used by simulation.py
│
└── YOLO/
    └── OT2-Computer-Vision/
        └── Trained Models_NAS/
            └── ckpt_best.pth        # Trained YOLO-NAS model weights (git-ignored)
```

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd autolabproject
pip install -r requirements.txt
```

`requirements.txt` lists every third-party package with minimum versions. See the file for notes on the `super-gradients` / numpy pin and how to install without the YOLO inference layer if you only need the campaign runner.

### 2. Start the camera server (on your computer, not the robot)

```bash
python camera/camera_server.py
```

The server listens on port 8080. Adjust `CAMERA_INDEX` in the file if your USB camera is not the default device.

### 3. Run the campaign

```bash
cd autolabproject
python experiments/experiment_runner.py
```

A browser window opens automatically with the **Setup Wizard**. Follow the five steps:

| Step | What happens |
|------|-------------|
| 1 — Connection | Checks OT-2 robot + camera server are reachable |
| 2 — Configure | Choose which parameters to optimise, set fixed values for the rest, configure stopping criteria |
| 3 — Deck Layout | Visual guide showing where to place each labware item |
| 4 — Calibration | Verifies deck and pipette calibration via the robot API |
| 5 — Review & Start | Summary of all settings → launches the campaign |

After clicking **Start Campaign**, the browser switches to the live **Campaign Dashboard**.

### 4. Test without hardware (dry run)

```bash
python experiments/experiment_runner.py --dry-run
```

Enables Dry Run in the wizard automatically. The robot and camera are skipped; accuracy scores are random. Use this to test the full pipeline before connecting hardware.

### 5. Skip the wizard (headless)

```bash
python experiments/experiment_runner.py --no-wizard              # live, settings from campaign_config.py
python experiments/experiment_runner.py --no-wizard --dry-run    # dry run, no browser
```

---

## Configuration

All campaign settings live in **`experiments/campaign_config.py`**. Edit this file directly, or override via the Setup Wizard at runtime.

```python
# Which parameters to optimise and their ranges
FACTORS = [
    Factor("aspirate_speed", "continuous",  1.0, 200.0),   # µL/s
    Factor("dispense_speed", "continuous",  1.0, 200.0),   # µL/s
    Factor("air_gap",        "categorical", 0.0,   1.0, levels=[0.0, 1.0]),
    Factor("blow_out",       "categorical", 0.0,   1.0, levels=[0.0, 1.0]),
]

N_INITIAL          = 5      # LHS points before Bayesian optimisation begins
MAX_ITERATIONS     = 20     # Hard cap on total experiments
CONVERGENCE_TOL    = 1.0    # Stop if best accuracy improves < 1% over the window
CONVERGENCE_WINDOW = 5      # Number of consecutive BO iterations to check

ROBOT_IP           = None   # None = auto-detect across known IPs
CAMERA_SERVER_IP   = "169.254.84.3"
CAMERA_SERVER_PORT = 8080
```

---

## OT-2 Deck Layout

```
  ┌────────┬────────┬────────┐
  │ Slot 7 │ Slot 8 │ Slot 9 │  ← back (away from you)
  │  CAM   │  ---   │  ---   │
  ├────────┼────────┼────────┤
  │ Slot 4 │ Slot 5 │ Slot 6 │
  │  RES   │  ---   │  ---   │
  ├────────┼────────┼────────┤
  │ Slot 1 │ Slot 2 │ Slot 3 │  ← front (closest to you)
  │  TIP   │  PLT   │  ---   │
  └────────┴────────┴────────┘
```

| Slot | Labware | Purpose |
|------|---------|---------|
| 1 | `opentrons_96_filtertiprack_20ul` | 20 µL filter tip rack |
| 2 | `corning_96_wellplate_330ul` | Destination well plate |
| 4 | `agilent_1_reservoir_290ml` | Liquid source reservoir |
| 7 | `corning_96_wellplate_360ul_flat` | Camera imaging position |

The pipette (`p20_multi_gen2`) is on the **left mount**.

---

## Campaign Dashboard

Open automatically at `http://localhost:9999/` when a campaign starts. Refreshes every 3 seconds.

**Controls:**

| Button | Action |
|--------|--------|
| Pause | Finish the current image capture, then wait |
| Resume | Continue from a paused state |
| Stop After Iteration | Finish the current experiment, then stop cleanly |
| Stop Now | Immediately halt the OT-2 mid-run |

**Sections:**

- **Stat cards** — experiments run, best accuracy, current phase, mean, std dev, latest EI
- **Next Suggested Parameters** — what the optimiser will try next
- **Stopping Criteria** — live status of convergence and cap checks
- **Best Parameters Found** — current best settings
- **Robot Status** — current step, OT-2 run status, iteration number
- **Camera Live View** — most recent tip image, updates every 2 s (shows "Dry Run" placeholder in dry run mode)
- **Accuracy Chart** — scatter plot of all iterations with best-so-far line
- **Experiment Log** — full table of every iteration with parameters, accuracy, EI, protocol filename

---

## Results

After each iteration, a row is appended to **`experiments/results/results.csv`**:

```
iteration, timestamp, phase, aspirate_speed, dispense_speed, air_gap, blow_out,
accuracy_%, ei, best_so_far_%, protocol_file
```

At campaign end:
- `experiments/results/convergence_plot.png` — accuracy vs. iteration chart
- `experiments/results/campaign_log.csv` — optimiser internal state log
- `experiments/protocols/iter_NNN_asp7.6_disp26_ag-off_bo-on_20260326_143022.py` — one protocol file per iteration (git-ignored)

### Protocol filename format

```
iter_003_asp174.4_disp26_ag-off_bo-on_20260326_143022.py
│        │        │       │      │     └── timestamp
│        │        │       │      └── blow_out on/off
│        │        │       └── air_gap on/off
│        │        └── dispense speed (µL/s, 1 decimal)
│        └── aspirate speed (µL/s, 1 decimal)
└── iteration number
```

---

## Liquid Analysis

The analyser (`camera/analysis_script/liquid_analysis.py`) scores each iteration by comparing captured tip images to expected states:

| Capture moment | Expected | Meaning |
|----------------|----------|---------|
| `before_aspirate` | No liquid | Tip should be empty |
| `after_aspirate` | Liquid present | Tip should be loaded |
| `before_dispense` | Liquid present | Liquid still in tip |
| `after_dispense` | No liquid | Clean dispense — the key quality metric |

**Accuracy = % of images where the detected state matches the expected state.**

Two detection methods are combined:
1. **YOLO-NAS** — detects liquid bounding boxes and fill ratio (`YOLO/OT2-Computer-Vision/Trained Models_NAS/ckpt_best.pth`)
2. **HSV colour segmentation** — detects green liquid by hue/saturation thresholding (primary signal for green dye)

---

## Running a Single Protocol (without campaign)

Use the universal runner to upload and run any OT-2 protocol file:

```bash
python OT2_operation/run_protocol.py path/to/protocol.py             # live run
python OT2_operation/run_protocol.py path/to/protocol.py --simulate  # simulate on robot
```

The runner auto-parses runtime parameters from the protocol file, prompts for values (with defaults), checks calibration, and streams live run status.

---

## Network Setup

The OT-2, your computer, and the camera must all be on the same network segment.

| Device | Default IP | Port | Purpose |
|--------|-----------|------|---------|
| OT-2 (USB) | `169.254.84.3` | `31950` | Robot REST API |
| OT-2 (Wi-Fi) | `172.26.4.16` or `172.26.4.17` | `31950` | Robot REST API |
| Camera server | `169.254.84.3` (same machine as OT-2 USB host) | `8080` | Image capture |

The robot IP is auto-detected by trying each known address. Override `ROBOT_IP` in `campaign_config.py` to pin a specific address.

---

## Dependencies

All dependencies are declared in **`requirements.txt`**. Install with:

```bash
pip install -r requirements.txt
```

| Package | Version | Use |
|---------|---------|-----|
| `numpy` | ==1.26.4 | Array operations; pinned `<2` for `super-gradients` compatibility |
| `scipy` | >=1.11 | Gaussian Process optimisation, Latin Hypercube Sampling |
| `scikit-learn` | >=1.3 | `GaussianProcessRegressor`, Matern kernel |
| `matplotlib` | >=3.7 | Convergence plot saved to PNG (Agg backend — no display required) |
| `opencv-python` | >=4.8 | Camera capture (`camera_server.py`) and image processing |
| `super-gradients` | >=3.7 | YOLO-NAS model loading for liquid detection |

Everything else (`http.server`, `pathlib`, `threading`, `urllib`, etc.) is Python standard library — no install needed.

---

## Git-Ignored Files

| Path | Reason |
|------|--------|
| `experiments/protocols/` | Generated per-run; tracked externally |
| `*.pth` | Large model weights |
| `*.jpg`, `*.mp4` | Captured images and videos |
| `*.rar` | Compressed archives |
