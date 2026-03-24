"""
OT-2 Transfer Protocol with Automatic Camera Capture

Transfers 20 uL from reservoir (slot 4) to well plate columns (slot 7),
taking a photo after every pick_up_tip, aspirate, dispense, and drop_tip.

The pipette moves to the camera slot and pauses before each capture
so the image is clear (no vibration).

Setup:
  1. Start camera_server.py on your computer first
  2. Upload this protocol to the OT-2 via the Opentrons App
  3. Set runtime parameters as needed before starting the run
"""

from opentrons import protocol_api
import json

metadata = {
    "protocolName": "Project - Auto Capture",
    "author": "AutoLab",
    "description": "Transfer with automatic camera capture after each action.",
}
requirements = {"robotType": "OT-2", "apiLevel": "2.18"}


# ── Runtime Parameters ─────────────────────────────────────────────────────
def add_parameters(parameters: protocol_api.Parameters):
    parameters.add_str(
        variable_name="camera_server_ip",
        display_name="Camera Server IP",
        default="169.254.84.3",
        choices=[
            {"display_name": "169.254.84.3", "value": "169.254.84.3"},
            {"display_name": "172.26.4.16", "value": "172.26.4.16"},
            {"display_name": "172.26.4.17", "value": "172.26.4.17"},
        ],
    )
    parameters.add_int(
        variable_name="camera_server_port",
        display_name="Camera Server Port",
        default=8080,
        minimum=1024,
        maximum=65535,
    )
    parameters.add_int(
        variable_name="num_columns",
        display_name="Number of Columns",
        description="How many columns to dispense into (1-12)",
        default=12,
        minimum=1,
        maximum=12,
    )
    parameters.add_int(
        variable_name="transfer_volume",
        display_name="Transfer Volume (uL)",
        default=20,
        minimum=1,
        maximum=20,
    )
    parameters.add_bool(
        variable_name="capture_enabled",
        display_name="Enable Camera Capture",
        default=True,
    )
    parameters.add_int(
        variable_name="settle_seconds",
        display_name="Settle Time (seconds)",
        description="Wait time before capture for clear image",
        default=2,
        minimum=1,
        maximum=10,
    )
    parameters.add_int(
        variable_name="extra_wait_seconds",
        display_name="Extra Wait After Capture (seconds)",
        default=3,
        minimum=0,
        maximum=10,
    )


# ── Camera Capture Helper ──────────────────────────────────────────────────
def make_capture_fn(protocol: protocol_api.ProtocolContext):
    base_url = (
        f"http://{protocol.params.camera_server_ip}"
        f":{protocol.params.camera_server_port}"
    )
    enabled = protocol.params.capture_enabled

    def capture(action: str, details: str = ""):
        if not enabled:
            return
        if protocol.is_simulating():
            protocol.comment(f"[SIM] Would capture: {action} {details}")
            return

        import urllib.request
        import urllib.error

        payload = json.dumps({"action": action, "details": details}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/capture",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
                protocol.comment(f"[CAPTURE] {action}: {result.get('image_path', '?')}")
        except urllib.error.URLError as e:
            protocol.comment(f"[CAPTURE ERROR] {action}: {e}")

    return capture


# ── Main Protocol ──────────────────────────────────────────────────────────
def run(protocol: protocol_api.ProtocolContext):
    capture = make_capture_fn(protocol)
    params = protocol.params

    settle = params.settle_seconds
    extra = params.extra_wait_seconds
    volume = params.transfer_volume
    num_cols = params.num_columns

    # ── Load Labware ───────────────────────────────────────────────────
    tip_rack = protocol.load_labware(
        "opentrons_96_filtertiprack_20ul", "1",
    )
    reservoir = protocol.load_labware(
        "agilent_1_reservoir_290ml", "4",
    )
    well_plate = protocol.load_labware(
        "corning_96_wellplate_330ul", "2",
    )
    # Camera position: dummy labware in slot 7 for photo
    camera_spot = protocol.load_labware(
        "corning_96_wellplate_360ul_flat", "7",
    )

    # ── Load Pipette ───────────────────────────────────────────────────
    pipette = protocol.load_instrument(
        "p20_multi_gen2", mount="left", tip_racks=[tip_rack],
    )
    pipette.configure_nozzle_layout(protocol_api.ALL, start="A1")

    # ── Flow rates from original liquid class ──────────────────────────
    ASPIRATE_RATE = 7.6    # uL/s
    DISPENSE_RATE = 26.0   # uL/s
    BLOWOUT_RATE = 31.0    # uL/s

    # ── Helper: move to camera, wait, capture ──────────────────────────
    def move_and_capture(action, details=""):
        pipette.move_to(camera_spot["A1"].top(20))
        protocol.delay(seconds=settle)
        capture(action, details)
        protocol.delay(seconds=extra)

    # ── Define columns to process ──────────────────────────────────────
    columns = [f"A{i}" for i in range(1, num_cols + 1)]

    # ── Liquid ─────────────────────────────────────────────────────────
    liquid = protocol.define_liquid("Water", display_color="#25b3ffff")
    reservoir.load_liquid(wells=["A1"], liquid=liquid, volume=290000)

    # ── PROTOCOL STEPS ─────────────────────────────────────────────────
    protocol.comment("=== Starting transfer protocol ===")

    # Pick up tip (once for all columns, matching original)
    pipette.pick_up_tip()
    move_and_capture("pick_up_tip", "column_A1")

    # Pre-wet the tip (matching original liquid class pre_wet=True)
    pipette.aspirate(volume, reservoir["A1"].bottom(z=1), rate=ASPIRATE_RATE / pipette.flow_rate.aspirate)
    pipette.dispense(volume, reservoir["A1"].bottom(z=1), rate=DISPENSE_RATE / pipette.flow_rate.dispense)

    # Transfer to each column
    for col_idx, col in enumerate(columns):
        protocol.comment(f"--- Column {col} ({col_idx + 1}/{num_cols}) ---")

        # Aspirate from reservoir
        pipette.aspirate(
            volume,
            reservoir["A1"].bottom(z=1),
            rate=ASPIRATE_RATE / pipette.flow_rate.aspirate,
        )
        move_and_capture("aspirate", f"col_{col}_from_reservoir")

        # Dispense into well plate
        pipette.dispense(
            volume,
            well_plate[col].bottom(z=0.3),
            rate=DISPENSE_RATE / pipette.flow_rate.dispense,
        )

        # Blowout at destination (matching original liquid class)
        pipette.blow_out(well_plate[col])

        move_and_capture("dispense", f"col_{col}_to_wellplate")

    # Drop tip
    pipette.drop_tip()
    move_and_capture("drop_tip", "final")

    protocol.comment("=== Protocol complete ===")
