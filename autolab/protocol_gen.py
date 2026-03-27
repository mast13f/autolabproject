"""
Protocol Generator
==================
Generates a self-contained, runnable OT-2 protocol .py file for each
iteration of the optimization campaign. The optimized parameters are baked
in as constants so the file is a complete record of what was run.

Naming: iter_001_asp7.6_disp26.0_ag-off_bo-on_20260326_143022.py

Rules:
  - Continuous values are rounded to 1 decimal place.
  - Boolean parameters (air_gap, blow_out) are shown as "on" / "off".
  - Timestamp goes at the end so iteration number stays visually prominent.
"""

import time
from pathlib import Path

# Parameters treated as boolean (0 = off, non-zero = on)
_BOOL_PARAMS = {"air_gap", "blow_out"}

# Short keys for each known parameter
_SHORT_KEY = {
    "aspirate_speed": "asp",
    "dispense_speed": "disp",
    "air_gap":        "ag",
    "blow_out":       "bo",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_value(name: str, value) -> str:
    """Format a parameter value for the filename."""
    if name in _BOOL_PARAMS:
        return "on" if float(value) else "off"
    # Continuous: round to 1 decimal, strip trailing zero only if it's .0
    rounded = round(float(value), 1)
    return f"{rounded:.1f}".rstrip("0").rstrip(".") or "0"


def _param_summary(params: dict) -> str:
    """One-line summary of parameters for the filename."""
    parts = []
    for k, v in params.items():
        key = _SHORT_KEY.get(k, k[:4])
        parts.append(f"{key}{_format_value(k, v)}")
    return "_".join(parts)


# ── Generator ────────────────────────────────────────────────────────────────

def generate_protocol(
    params: dict,
    iteration: int,
    protocols_dir: Path,
    num_columns: int = 12,
    transfer_volume: int = 20,
    camera_ip: str = "169.254.84.3",
    camera_port: int = 8080,
    settle_seconds: int = 2,
    camera_height_mm: int = 1,
) -> Path:
    """
    Generate a runnable OT-2 protocol file with the given parameters baked in.

    Args:
        params:           Dict of factor_name → value from the optimizer.
        iteration:        Current iteration number (used in filename + metadata).
        protocols_dir:    Directory to save the file in.
        num_columns:      Number of plate columns to transfer into.
        transfer_volume:  µL to transfer per column.
        camera_ip:        Camera server IP (default for the runtime param).
        camera_port:      Camera server port.
        settle_seconds:   Wait time before image capture.
        camera_height_mm: Height above slot 7 for the camera photo spot.

    Returns:
        Path to the saved protocol file.
    """
    protocols_dir = Path(protocols_dir)
    protocols_dir.mkdir(parents=True, exist_ok=True)

    # Extract known factors (fall back to sensible defaults)
    aspirate_speed = float(params.get("aspirate_speed", 7.6))
    dispense_speed = float(params.get("dispense_speed", 26.0))
    air_gap        = int(round(float(params.get("air_gap",  0))))
    blow_out       = int(round(float(params.get("blow_out", 1))))

    summary = _param_summary(params)
    ts      = time.strftime("%Y%m%d_%H%M%S")
    fname   = f"iter_{iteration:03d}_{summary}_{ts}.py"
    fpath   = protocols_dir / fname

    # Code snippets that change based on parameters
    air_gap_after_aspirate = (
        "        pipette.air_gap(2)  # prevent dripping\n"
        if air_gap else ""
    )
    air_gap_before_dispense = (
        "        pipette.dispense(2, dest.bottom(z=0.3))  # release air gap\n"
        if air_gap else ""
    )
    blow_out_code = (
        "        pipette.blow_out(dest)  # expel any residual\n"
        if blow_out else ""
    )

    code = f'''"""
AutoLab Experiment Protocol — Iteration {iteration}
=====================================================
Generated : {time.strftime("%Y-%m-%d %H:%M:%S")}
Parameters: {params}

Deck layout
-----------
  Slot  1 : Tip rack         (opentrons_96_filtertiprack_20ul)
  Slot  2 : Well plate       (corning_96_wellplate_330ul)
  Slot  4 : Source reservoir (agilent_1_reservoir_290ml)
  Slot  7 : Camera reservoir (agilent_1_reservoir_290ml — pipette moves here for photos)
  TRASH   : Fixed top-right
"""

from opentrons import protocol_api
import json

metadata = {{
    "protocolName": "AutoLab Iter {iteration} — {summary}",
    "author": "AutoLab Experiment Runner",
    "description": "iter={iteration} | asp={aspirate_speed} disp={dispense_speed} ag={air_gap} bo={blow_out}",
}}
requirements = {{"robotType": "OT-2", "apiLevel": "2.18"}}

# ── Optimized parameters (set by Bayesian optimizer) ─────────────────────────
ASPIRATE_SPEED = {aspirate_speed}   # µL/s
DISPENSE_SPEED = {dispense_speed}   # µL/s
AIR_GAP        = {bool(air_gap)}         # add 2 µL air gap after aspirate
BLOW_OUT       = {bool(blow_out)}        # blow out at destination after dispense

# ── Fixed parameters ──────────────────────────────────────────────────────────
NUM_COLUMNS     = {num_columns}
TRANSFER_VOLUME = {transfer_volume}


# ── Runtime parameters (camera settings — rarely need changing) ───────────────
def add_parameters(parameters: protocol_api.Parameters):
    parameters.add_str(
        variable_name="camera_server_ip",
        display_name="Camera Server IP",
        default="{camera_ip}",
        choices=[
            {{"display_name": "{camera_ip}", "value": "{camera_ip}"}},
            {{"display_name": "169.254.84.3",  "value": "169.254.84.3"}},
            {{"display_name": "172.26.4.16",   "value": "172.26.4.16"}},
            {{"display_name": "172.26.4.17",   "value": "172.26.4.17"}},
        ],
    )
    parameters.add_int(
        variable_name="camera_server_port",
        display_name="Camera Server Port",
        default={camera_port},
        minimum=1024, maximum=65535,
    )
    parameters.add_bool(
        variable_name="capture_enabled",
        display_name="Enable Camera Capture",
        default=True,
    )
    parameters.add_int(
        variable_name="settle_seconds",
        display_name="Settle Time (s)",
        default={settle_seconds}, minimum=1, maximum=10,
    )
    parameters.add_int(
        variable_name="camera_height_mm",
        display_name="Camera Height (mm)",
        default={camera_height_mm}, minimum=-10, maximum=80,
    )


# ── Camera capture helper ─────────────────────────────────────────────────────
def _make_capture(protocol: protocol_api.ProtocolContext):
    base_url = (
        f"http://{{protocol.params.camera_server_ip}}"
        f":{{protocol.params.camera_server_port}}"
    )
    enabled = protocol.params.capture_enabled

    def capture(action: str, details: str = ""):
        if not enabled:
            return
        if protocol.is_simulating():
            protocol.comment(f"[SIM] capture: {{action}} {{details}}")
            return
        import urllib.request, urllib.error
        payload = json.dumps({{"action": action, "details": details}}).encode()
        req = urllib.request.Request(
            f"{{base_url}}/capture", data=payload,
            headers={{"Content-Type": "application/json"}}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                result = json.loads(r.read().decode())
                protocol.comment(f"[CAP] {{action}}: {{result.get('image_path','?')}}")
        except urllib.error.URLError as e:
            protocol.comment(f"[CAP ERROR] {{action}}: {{e}}")

    return capture


# ── Main protocol ─────────────────────────────────────────────────────────────
def run(protocol: protocol_api.ProtocolContext):
    capture = _make_capture(protocol)
    settle  = protocol.params.settle_seconds
    cam_h   = protocol.params.camera_height_mm

    # Load labware
    tip_rack    = protocol.load_labware("opentrons_96_filtertiprack_20ul", "1")
    well_plate  = protocol.load_labware("corning_96_wellplate_330ul", "2")
    reservoir   = protocol.load_labware("agilent_1_reservoir_290ml", "4")
    cam_spot    = protocol.load_labware("agilent_1_reservoir_290ml", "7")  # camera films here

    # Load pipette
    pipette = protocol.load_instrument(
        "p20_multi_gen2", mount="left", tip_racks=[tip_rack]
    )
    pipette.configure_nozzle_layout(protocol_api.ALL, start="A1")

    # Convert absolute µL/s to Opentrons flow-rate ratio
    asp_ratio  = ASPIRATE_SPEED  / pipette.flow_rate.aspirate
    disp_ratio = DISPENSE_SPEED / pipette.flow_rate.dispense

    def move_and_capture(action, details=""):
        pipette.move_to(cam_spot["A1"].top(cam_h))
        protocol.delay(seconds=settle)
        capture(action, details)

    tip_cols  = [tip_rack[f"A{{i}}"]   for i in range(1, NUM_COLUMNS + 1)]
    dest_cols = [well_plate[f"A{{i}}"] for i in range(1, NUM_COLUMNS + 1)]

    protocol.comment(
        f"=== AutoLab Iter {iteration}: "
        f"asp={{ASPIRATE_SPEED}} disp={{DISPENSE_SPEED}} "
        f"air_gap={{AIR_GAP}} blow_out={{BLOW_OUT}} ==="
    )

    for idx, (tip_well, dest) in enumerate(zip(tip_cols, dest_cols)):
        col = f"A{{idx + 1}}"
        protocol.comment(f"--- Column {{col}} ---")

        # Pick up tips and pre-wet
        pipette.pick_up_tip(tip_well)
        pipette.aspirate(TRANSFER_VOLUME, reservoir["A1"].bottom(z=1), rate=asp_ratio)
        pipette.dispense(TRANSFER_VOLUME, reservoir["A1"].bottom(z=1), rate=disp_ratio)

        # BEFORE ASPIRATE — empty tip baseline image
        move_and_capture("before_aspirate", f"col_{{col}}_empty")

        # ASPIRATE
        pipette.aspirate(TRANSFER_VOLUME, reservoir["A1"].bottom(z=1), rate=asp_ratio)
{air_gap_after_aspirate}
        # AFTER ASPIRATE — liquid-filled tips
        move_and_capture("after_aspirate", f"col_{{col}}_filled")

        # BEFORE DISPENSE
        move_and_capture("before_dispense", f"col_{{col}}_pre_dispense")

        # DISPENSE
{air_gap_before_dispense}        pipette.dispense(TRANSFER_VOLUME, dest.bottom(z=0.3), rate=disp_ratio)
{blow_out_code}
        # AFTER DISPENSE — should be empty (key quality metric)
        move_and_capture("after_dispense", f"col_{{col}}_post_dispense")

        pipette.drop_tip()

    protocol.comment("=== Protocol complete ===")
'''

    fpath.write_text(code, encoding="utf-8")
    print(f"  [PROTO] {fname}")
    return fpath
