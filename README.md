# AutoLab — Closed-Loop Liquid Handling Optimiser

AutoLab is a self-driving laboratory system that optimises OT-2 pipette parameters using Bayesian optimisation. It runs experiments, captures images, analyses liquid detection accuracy, and feeds results back to the optimiser. Between experiments the campaign pauses for operator confirmation so you can replace labware or refill reservoirs before continuing.

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
├── autolab/                         # Main campaign orchestration package
│   ├── runner.py                    # Entry point — runs the full campaign loop
│   ├── config.py                    # User-editable settings (factors, limits, IPs)
│   ├── dashboard.py                 # Live HTTP dashboard with pause/stop controls
│   ├── wizard.py                    # 5-step web wizard that runs before the campaign
│   └── protocol_gen.py              # Generates a .py protocol file per iteration
│
├── robot/                           # OT-2 robot control
│   ├── controller.py                # OT-2 HTTP REST client (no Opentrons App needed)
│   └── run_protocol.py              # Universal protocol runner (any .py file)
│
├── camera/                          # Camera capture and image analysis
│   ├── server.py                    # HTTP server that captures images from USB camera
│   ├── live_view.py                 # Live preview utility for camera positioning
│   └── analysis.py                  # YOLO-NAS + HSV liquid detection pipeline
│
├── optimizer/                       # Bayesian optimisation engine
│   ├── core.py                      # DOEOptimiser class (Gaussian Process + LHS + EI)
│   └── _sim_dashboard.py            # Internal static HTML dashboard for the optimiser
│
├── models/                          # Trained model weights
│   └── ckpt_best.pth                # YOLO-NAS weights (git-ignored — large binary)
│
├── samples/                         # Example OT-2 protocols with camera capture
│   ├── good_protocol_water.py       # Reference water-transfer protocol
│   └── protocol_sample.py           # Minimal template
│
├── protocols/                       # Generated per-iteration protocol files (git-ignored)
│
├── results/                         # Campaign output (git-ignored)
│   ├── results.csv                  # Full log: parameters, accuracy, EI per iteration
│   ├── campaign_log.csv             # Optimiser internal log
│   ├── convergence_plot.png         # Accuracy plot saved at campaign end
│   └── iter_NNN/                    # Per-iteration folder
│       ├── images/                  # Captured tip images
│       └── analysis/                # Liquid analysis output (CSV + annotated images)
│
└── captured_images/                 # Live camera captures (git-ignored)
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

### 2. Start the camera server (automatic)

The camera server (`camera/server.py`) **starts automatically** when you launch a live campaign. You do not need to run it manually — AutoLab launches it as a subprocess with the exact settings from the Setup Wizard and shuts it down when the campaign ends.

If the server is already running (e.g. you started it manually for testing), AutoLab detects this and reuses it instead of launching a second instance.

**Manual launch** (for testing the camera independently):

```bash
python camera/server.py --camera-index 1 --port 8080 --save-dir /path/to/autolabproject/captured_images
```

| Flag | Default | Description |
|------|---------|-------------|
| `--camera-index` | `1` | OpenCV device index (`0` = built-in webcam, `1` = first USB camera) |
| `--port` | `8080` | HTTP port the OT-2 protocol connects to |
| `--save-dir` | `./captured_images` | Where images are saved |

### 3. Run the campaign

```bash
cd autolabproject
python -m autolab.runner
```

A browser window opens automatically with the **Setup Wizard**. Follow the five steps:

| Step | What happens |
|------|-------------|
| 1 — Connection | Enter robot and camera server addresses, run a connection check, toggle Dry Run if needed |
| 2 — Configure | Choose which parameters to optimise, set fixed values for the rest, configure stopping criteria |
| 3 — Deck Layout | Visual guide showing where to place each labware item |
| 4 — Calibration | Verifies deck and pipette calibration via the robot API |
| 5 — Review & Start | Summary of all settings → launches the campaign |

After clicking **Start Campaign**, the browser switches to the live **Campaign Dashboard**.

### 4. Skip the wizard (headless)

```bash
python -m autolab.runner --no-wizard              # live, settings from autolab/config.py
python -m autolab.runner --no-wizard --dry-run    # dry run, no browser
```

---

## Configuration

All campaign settings live in **`autolab/config.py`**. Edit this file directly, or override via the Setup Wizard at runtime.

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
  ┌─────────┬─────────┬─────────┐
  │ Slot 10 │ Slot 11 │  TRASH  │  ← top row (back of robot)
  │   ---   │   ---   │  fixed  │
  ├─────────┼─────────┼─────────┤
  │  Slot 7 │  Slot 8 │  Slot 9 │
  │   CAM   │   ---   │   ---   │
  ├─────────┼─────────┼─────────┤
  │  Slot 4 │  Slot 5 │  Slot 6 │
  │   SRC   │   ---   │   ---   │
  ├─────────┼─────────┼─────────┤
  │  Slot 1 │  Slot 2 │  Slot 3 │  ← front row (closest to you)
  │   TIP   │   PLT   │   ---   │
  └─────────┴─────────┴─────────┘
```

| Slot | Labware | Purpose |
|------|---------|---------|
| 1 | `opentrons_96_filtertiprack_20ul` | 20 µL filter tip rack |
| 2 | `corning_96_wellplate_330ul` | Destination well plate |
| 4 | `agilent_1_reservoir_290ml` | Source liquid reservoir |
| 7 | `agilent_1_reservoir_290ml` | Camera reservoir — pipette moves here for every image; camera films tips from above |
| TRASH | *(fixed)* | Built-in trash bin, top-right corner |

The pipette (`p20_multi_gen2`) is on the **left mount**.

---

## Campaign Dashboard

Open automatically at `http://localhost:9999/` when a campaign starts. Refreshes every 3 seconds.

**Controls:**

| Button | Action |
|--------|--------|
| Pause | Finish the current image capture, then wait |
| Resume | Continue from a paused state |
| Confirm & Run Next Experiment | Appears after each completed experiment — click to start the next one |
| Stop After Iteration | Finish the current experiment, then stop cleanly (also works while awaiting confirmation) |
| Stop Now | Immediately halt the OT-2 mid-run |

**Inter-experiment confirmation gate:**

After every completed experiment the campaign enters an **Awaiting Confirmation** state. The dashboard displays a prominent panel reminding you to replace labware, refill reservoirs, or make any deck changes. The next experiment does **not** start — and its protocol is **not** generated or uploaded — until you click **Confirm & Run Next Experiment**. The first experiment in a campaign starts immediately; confirmation is required only between experiments.

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
1. **YOLO-NAS** — detects liquid bounding boxes and fill ratio (`YOLO/OT2-Computer-Vision/Trained Models_NAS/ckpt_best.pth`).
   Model trained and provided by [BDD-G/OT2-Computer-Vision](https://github.com/BDD-G/OT2-Computer-Vision).
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
| `protocols/` | Generated per-run; tracked externally |
| `results/` | Campaign output data |
| `captured_images/` | Raw camera captures |
| `*.pth` | Large model weights |
| `*.jpg`, `*.mp4` | Captured images and videos |
| `*.rar` | Compressed archives |

---

## Roadmap / Next Steps

### 1 — Destination labware selection (well plate vs reservoir)

Currently Slot 2 is always a `corning_96_wellplate_330ul`. A future Setup Wizard option would let the user choose:

| Mode | Slot 2 labware | Use case |
|------|---------------|----------|
| **Well plate** *(current)* | `corning_96_wellplate_330ul` | Dispense into 96 individual wells, one column per tip-set |
| **Reservoir** | `agilent_1_reservoir_290ml` | Bulk transfer into a single trough (e.g. for reagent prep or waste collection) |

This requires:
- A new toggle in the Setup Wizard Step 2 ("Destination type: Well plate / Reservoir")
- A second sample protocol (`samples/reservoir_transfer.py`) where `dest_cols = [dest_res["A1"]] * NUM_COLUMNS` instead of iterating over plate columns
- The protocol generator to branch on the selected destination type and load the correct labware in Slot 2

### 2 — Plate reader verification

After each dispense iteration, cross-validate the camera-based accuracy score against absorbance readings from a plate reader:

- **Goal:** confirm that the YOLO/HSV liquid detection score correlates with ground-truth volume via OD measurement
- **Integration points:**
  - Add a post-run step in `autolab/runner.py` that triggers a plate reader measurement (via USB/serial or HTTP if the reader has an API)
  - Log both the camera accuracy score and the plate reader OD value per iteration in `results/results.csv`
  - Display both metrics side-by-side in the Campaign Dashboard
  - Optionally use the plate reader OD as the primary optimisation objective instead of (or in addition to) the camera score
