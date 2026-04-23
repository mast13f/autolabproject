"""
Campaign Configuration
=======================
Edit this file to configure your optimization campaign.

Factors:       which parameters to optimize and their ranges
Optimizer:     how the Bayesian optimization behaves
Protocol:      liquid handling settings applied to every run
Robot/Camera:  connection settings
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent   # project root (autolabproject/)

from optimizer import Factor          # noqa: E402  (optimizer/ package at root)


# ── Factors to optimize ──────────────────────────────────────────────────────
# Ranges calibrated for OT-2 p20 multi-channel pipette.
FACTORS = [
    Factor("aspirate_speed", "continuous",  0.0, 31.0),
    Factor("dispense_speed", "continuous",  1.0, 31.0),
    Factor("air_gap",        "categorical", 0.0,  1.0, levels=[0.0, 1.0]),
    Factor("blow_out",       "categorical", 0.0,  1.0, levels=[0.0, 1.0]),
]

# ── Liquid settings ─────────────────────────────────────────────────────────
LIQUID_TYPE        = "water"
CONCENTRATION_PCT  = 100.0

# ── Image analysis settings ─────────────────────────────────────────────────
IMAGE_THRESHOLD    = 12       # Binary threshold for image subtraction
DYE_COLOR          = "blue"   # Dye colour bias: "red", "blue", or "green"

# ── Optimizer settings ───────────────────────────────────────────────────────
N_INITIAL          = 5      # Latin Hypercube initial points before Bayesian opt kicks in
MAX_ITERATIONS     = 20     # Hard cap: stop after this many total experiments
XI                 = 0.01   # Exploration-exploitation balance (higher = more exploration)
CONVERGENCE_TOL    = 1.0    # Stop if best score improves < this % over window
CONVERGENCE_WINDOW = 5      # Number of consecutive BO iterations to check for convergence
RANDOM_SEED        = 42

# ── Protocol / liquid handling settings ──────────────────────────────────────
NUM_COLUMNS     = 12    # Columns to transfer per run (1–12)
TRANSFER_VOLUME = 20    # µL per transfer

# ── Robot / camera settings ───────────────────────────────────────────────────
ROBOT_IP           = "169.254.83.111" # Wired USB IP of the OT-2
CAMERA_SERVER_IP   = "127.0.0.1"     # IP of the computer running camera/server.py (localhost — server runs on this machine)
CAMERA_SERVER_PORT = 8080
CAMERA_INDEX       = 0                # USB camera device index (0 = default camera)
SETTLE_SECONDS     = 2                # Seconds to wait before capturing image
CAMERA_HEIGHT_MM   = 1                # Height above slot 7 plate top for photo

# ── Labware offsets (from Labware Position Check) ────────────────────────────
# These XYZ offsets are applied when creating each OT-2 run so the robot uses
# the calibrated positions.  Update these after re-running LPC in the
# Opentrons App.  Set to an empty list to skip (robot uses nominal positions).
LABWARE_OFFSETS = [
    {
        "definitionUri": "opentrons/agilent_1_reservoir_290ml/1",
        "location": {"slotName": "4"},
        "vector": {"x": -2.2, "y": -3.0, "z": 0.0},
    },
    {
        "definitionUri": "opentrons/corning_96_wellplate_330ul/1",
        "location": {"slotName": "2"},
        "vector": {"x": 0.7, "y": -3.3, "z": 10.1},
    },
    {
        "definitionUri": "opentrons/corning_96_wellplate_360ul_flat/1",
        "location": {"slotName": "7"},
        "vector": {"x": 3.0, "y": -1.9, "z": 18.5},
    },
    {
        "definitionUri": "opentrons/opentrons_96_filtertiprack_20ul/1",
        "location": {"slotName": "1"},
        "vector": {"x": 1.4, "y": -4.3, "z": -0.1},
    },
]

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_PORT = 9999     # Open http://localhost:9999 after starting the campaign

# ── Paths (all relative to project root) ──────────────────────────────────────
PROTOCOLS_DIR  = ROOT / "protocols"          # generated OT-2 protocol files
RESULTS_DIR    = ROOT / "results"            # CSV results, convergence plots
IMAGES_DIR     = ROOT / "captured_images"    # images saved by camera server
TRAINING_CSV   = str(ROOT / "results" / "training_data.csv")  # cross-run training log
