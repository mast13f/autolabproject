"""
OT-2 Water Transfer Protocol with Camera Capture

Adapted from Protocol Designer "optimalprojectprotocolwater.py".
Transfers 20 uL from reservoir (slot 4) to well plate (slot 2),
taking photos before and after every aspirate and dispense.

Deck layout:
  Slot 1: Tip rack (opentrons_96_filtertiprack_20ul)
  Slot 2: Well plate (corning_96_wellplate_330ul)
  Slot 4: Reservoir (agilent_1_reservoir_290ml)
  Slot 7: Camera position (dummy labware for photo spot)

Setup:
  1. Start camera_server.py on your computer first
  2. Upload this protocol to the OT-2 via the Opentrons App
  3. Set runtime parameters as needed before starting the run
"""

from opentrons import protocol_api
import json

metadata = {
    "protocolName": "Water Transfer - Camera",
    "author": "AutoLab",
    "description": "Water transfer with camera capture before/after aspirate and dispense.",
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
        variable_name="camera_height_mm",
        display_name="Camera Height (mm)",
        description="Height above slot 7 plate top for photo",
        default=1,
        minimum=-10,
        maximum=80,
    )
    # ── NEW: Air Gap toggle ────────────────────────────────────────────
    parameters.add_bool(
        variable_name="use_air_gap",
        display_name="Enable Air Gap",
        description="Aspirate a 2 uL air gap after liquid to prevent dripping",
        default=False,
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

    settle      = params.settle_seconds
    volume      = params.transfer_volume
    num_cols    = params.num_columns
    cam_height  = params.camera_height_mm
    use_air_gap = params.use_air_gap

    # Air gap volume in uL — fixed at 2 uL when enabled
    AIR_GAP_VOL = 2

    # ── Load Labware ───────────────────────────────────────────────────
    tip_rack = protocol.load_labware(
        "opentrons_96_filtertiprack_20ul", "1",
    )
    well_plate = protocol.load_labware(
        "corning_96_wellplate_330ul", "2",
    )
    reservoir = protocol.load_labware(
        "agilent_1_reservoir_290ml", "4",
    )
    camera_spot = protocol.load_labware(
        "corning_96_wellplate_360ul_flat", "7",
    )

    # ── Load Pipette ───────────────────────────────────────────────────
    pipette = protocol.load_instrument(
        "p20_multi_gen2", mount="left", tip_racks=[tip_rack],
    )
    pipette.configure_nozzle_layout(protocol_api.ALL, start="A1")

    # ── Flow rates ─────────────────────────────────────────────────────
    ASPIRATE_RATE = 11.82
    DISPENSE_RATE = 7.15
    BLOWOUT_RATE  = 31.0
    

    # ── Helper: move to camera spot, wait, capture ────────────────────
    def move_and_capture(action, details=""):
        pipette.move_to(camera_spot["A1"].top(cam_height))
        protocol.delay(seconds=settle)
        capture(action, details)

    # ── Columns to dispense into ──────────────────────────────────────
    dest_wells = [well_plate[f"A{i}"] for i in range(1, num_cols + 1)]
    tip_columns = [tip_rack[f"A{i}"] for i in range(1, num_cols + 1)]

    protocol.comment("=== Starting water transfer protocol with camera ===")

    # ── Transfer loop ─────────────────────────────────────────────────
    for col_idx, dest in enumerate(dest_wells):
        col_name = f"A{col_idx + 1}"
        protocol.comment(f"--- Column {col_name} ({col_idx + 1}/{num_cols}) ---")

        # Pick up tips
        pipette.pick_up_tip(tip_columns[col_idx])
        move_and_capture("pick_up_tip", f"col_{col_name}")

        # How much liquid to aspirate — reduced by air gap size so tip doesn't overflow
        aspirate_vol = volume - AIR_GAP_VOL if use_air_gap else volume

        # Pre-wet tip
        pipette.aspirate(aspirate_vol, reservoir["A1"].bottom(z=1), rate=ASPIRATE_RATE / pipette.flow_rate.aspirate)
        pipette.dispense(aspirate_vol, reservoir["A1"].bottom(z=1), rate=DISPENSE_RATE / pipette.flow_rate.dispense)

        # ── BEFORE ASPIRATE ──
        move_and_capture("before_aspirate", f"col_{col_name}_tips_empty")

        # ── ASPIRATE liquid ──
        pipette.aspirate(
            aspirate_vol,
            reservoir["A1"].bottom(z=1),
            rate=ASPIRATE_RATE / pipette.flow_rate.aspirate,
        )

        # ── AIR GAP (after aspirating liquid, before moving) ──
        # Air gap sits above the liquid in the tip, preventing dripping during movement
        if use_air_gap:
            pipette.air_gap(AIR_GAP_VOL)

        # ── AFTER ASPIRATE ──
        move_and_capture("after_aspirate", f"col_{col_name}_liquid_in_tips")

        # ── BEFORE DISPENSE ──
        move_and_capture("before_dispense", f"col_{col_name}_approaching_well")

        # ── DISPENSE into well ──
        # Dispense aspirate_vol only — Opentrons handles the air gap automatically
        pipette.dispense(
            aspirate_vol,
            dest.bottom(z=0.3),
            rate=DISPENSE_RATE / pipette.flow_rate.dispense,
        )

        # Blow out at destination
        pipette.blow_out(dest)

        # ── AFTER DISPENSE ──
        move_and_capture("after_dispense", f"col_{col_name}_liquid_dispensed")

        # Drop tip
        pipette.drop_tip()
        move_and_capture("drop_tip", f"col_{col_name}")

    protocol.comment("=== Protocol complete ===")