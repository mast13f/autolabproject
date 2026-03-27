"""
AutoLab Experiment Protocol — Iteration 8
=====================================================
Generated : 2026-03-27 00:31:57
Parameters: {'aspirate_speed': 10.0, 'dispense_speed': 102.93, 'air_gap': 1, 'blow_out': 0}

Deck layout
-----------
  Slot 1 : Tip rack  (opentrons_96_filtertiprack_20ul)
  Slot 2 : Well plate (corning_96_wellplate_330ul)
  Slot 4 : Reservoir  (agilent_1_reservoir_290ml)
  Slot 7 : Camera position (corning_96_wellplate_360ul_flat)
"""

from opentrons import protocol_api
import json

metadata = {
    "protocolName": "AutoLab Iter 8 — asp10.0_disp102.93_ag1_bo0",
    "author": "AutoLab Experiment Runner",
    "description": "iter=8 | asp=10.0 disp=102.93 ag=1 bo=0",
}
requirements = {"robotType": "OT-2", "apiLevel": "2.18"}

# ── Optimized parameters (set by Bayesian optimizer) ─────────────────────────
ASPIRATE_SPEED = 10.0   # µL/s
DISPENSE_SPEED = 102.93   # µL/s
AIR_GAP        = True         # add 2 µL air gap after aspirate
BLOW_OUT       = False        # blow out at destination after dispense

# ── Fixed parameters ──────────────────────────────────────────────────────────
NUM_COLUMNS     = 12
TRANSFER_VOLUME = 20


# ── Runtime parameters (camera settings — rarely need changing) ───────────────
def add_parameters(parameters: protocol_api.Parameters):
    parameters.add_str(
        variable_name="camera_server_ip",
        display_name="Camera Server IP",
        default="169.254.84.3",
        choices=[
            {"display_name": "169.254.84.3", "value": "169.254.84.3"},
            {"display_name": "169.254.84.3",  "value": "169.254.84.3"},
            {"display_name": "172.26.4.16",   "value": "172.26.4.16"},
            {"display_name": "172.26.4.17",   "value": "172.26.4.17"},
        ],
    )
    parameters.add_int(
        variable_name="camera_server_port",
        display_name="Camera Server Port",
        default=8080,
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
        default=2, minimum=1, maximum=10,
    )
    parameters.add_int(
        variable_name="camera_height_mm",
        display_name="Camera Height (mm)",
        default=1, minimum=-10, maximum=80,
    )


# ── Camera capture helper ─────────────────────────────────────────────────────
def _make_capture(protocol: protocol_api.ProtocolContext):
    base_url = (
        f"http://{protocol.params.camera_server_ip}"
        f":{protocol.params.camera_server_port}"
    )
    enabled = protocol.params.capture_enabled

    def capture(action: str, details: str = ""):
        if not enabled:
            return
        if protocol.is_simulating():
            protocol.comment(f"[SIM] capture: {action} {details}")
            return
        import urllib.request, urllib.error
        payload = json.dumps({"action": action, "details": details}).encode()
        req = urllib.request.Request(
            f"{base_url}/capture", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                result = json.loads(r.read().decode())
                protocol.comment(f"[CAP] {action}: {result.get('image_path','?')}")
        except urllib.error.URLError as e:
            protocol.comment(f"[CAP ERROR] {action}: {e}")

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
    cam_spot    = protocol.load_labware("corning_96_wellplate_360ul_flat", "7")

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

    tip_cols  = [tip_rack[f"A{i}"]   for i in range(1, NUM_COLUMNS + 1)]
    dest_cols = [well_plate[f"A{i}"] for i in range(1, NUM_COLUMNS + 1)]

    protocol.comment(
        f"=== AutoLab Iter 8: "
        f"asp={ASPIRATE_SPEED} disp={DISPENSE_SPEED} "
        f"air_gap={AIR_GAP} blow_out={BLOW_OUT} ==="
    )

    for idx, (tip_well, dest) in enumerate(zip(tip_cols, dest_cols)):
        col = f"A{idx + 1}"
        protocol.comment(f"--- Column {col} ---")

        # Pick up tips and pre-wet
        pipette.pick_up_tip(tip_well)
        pipette.aspirate(TRANSFER_VOLUME, reservoir["A1"].bottom(z=1), rate=asp_ratio)
        pipette.dispense(TRANSFER_VOLUME, reservoir["A1"].bottom(z=1), rate=disp_ratio)

        # BEFORE ASPIRATE — empty tip baseline image
        move_and_capture("before_aspirate", f"col_{col}_empty")

        # ASPIRATE
        pipette.aspirate(TRANSFER_VOLUME, reservoir["A1"].bottom(z=1), rate=asp_ratio)
        pipette.air_gap(2)  # prevent dripping

        # AFTER ASPIRATE — liquid-filled tips
        move_and_capture("after_aspirate", f"col_{col}_filled")

        # BEFORE DISPENSE
        move_and_capture("before_dispense", f"col_{col}_pre_dispense")

        # DISPENSE
        pipette.dispense(2, dest.bottom(z=0.3))  # release air gap
        pipette.dispense(TRANSFER_VOLUME, dest.bottom(z=0.3), rate=disp_ratio)

        # AFTER DISPENSE — should be empty (key quality metric)
        move_and_capture("after_dispense", f"col_{col}_post_dispense")

        pipette.drop_tip()

    protocol.comment("=== Protocol complete ===")
