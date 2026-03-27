"""
Campaign Configuration
=======================
Edit this file to configure your optimization campaign.

Factors:       which parameters to optimize and their ranges
Optimizer:     how the Bayesian optimization behaves
Protocol:      liquid handling settings applied to every run
Robot/Camera:  connection settings
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
# Add simulation/ directly so simulation.py can do `from dashboard import ...`
sys.path.insert(0, str(ROOT / "simulation"))

from simulation import Factor


# ── Factors to optimize ──────────────────────────────────────────────────────
# Add/remove factors here. Types: "continuous" or "categorical"
# For categorical, provide a `levels` list.
FACTORS = [
    Factor("aspirate_speed", "continuous",  1.0, 200.0),
    Factor("dispense_speed", "continuous",  1.0, 200.0),
    Factor("air_gap",        "categorical", 0.0,   1.0, levels=[0.0, 1.0]),
    Factor("blow_out",       "categorical", 0.0,   1.0, levels=[0.0, 1.0]),
]

# ── Optimizer settings ───────────────────────────────────────────────────────
N_INITIAL          = 5      # Latin Hypercube initial points before Bayesian opt kicks in
MAX_ITERATIONS     = 20     # Hard cap: stop after this many total experiments
XI                 = 0.01   # Exploration-exploitation balance (higher = more exploration)
CONVERGENCE_TOL    = 1.0    # Stop if best accuracy improves < this % over window
CONVERGENCE_WINDOW = 5      # Number of consecutive BO iterations to check for convergence
RANDOM_SEED        = 42

# ── Protocol / liquid handling settings ──────────────────────────────────────
NUM_COLUMNS     = 12    # Columns to transfer per run (1–12)
TRANSFER_VOLUME = 20    # µL per transfer

# ── Robot / camera settings ───────────────────────────────────────────────────
ROBOT_IP           = None             # None = auto-detect; or set e.g. "169.254.84.3"
CAMERA_SERVER_IP   = "169.254.84.3"   # IP of the computer running camera_server.py
CAMERA_SERVER_PORT = 8080
SETTLE_SECONDS     = 2                # Seconds to wait before capturing image
CAMERA_HEIGHT_MM   = 1                # Height above slot 7 plate for photo

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_PORT = 9999     # Open http://localhost:9999 after starting the campaign

# ── Paths (auto-derived, change if needed) ────────────────────────────────────
PROTOCOLS_DIR = Path(__file__).parent / "protocols"
RESULTS_DIR   = Path(__file__).parent / "results"
IMAGES_DIR    = ROOT / "captured_images"
WEIGHTS_PATH  = str(ROOT / "YOLO/OT2-Computer-Vision/Trained Models_NAS/ckpt_best.pth")
