"""
OT-2 Protocol with Automatic Camera Capture

This protocol automatically takes a picture (via the camera server running
on your computer) after pick_up_tip, aspirate, and dispense actions.

Setup:
  1. Start camera_server.py on your computer first
  2. Upload this protocol to the OT-2 via the Opentrons App
  3. Set runtime parameters as needed before starting the run

Network:
  Set CAMERA_SERVER_IP to your computer's IP address on the same network as the OT-2.
"""

from opentrons import protocol_api
import json

# ── Metadata ───────────────────────────────────────────────────────────────
metadata = {
    "protocolName": "Auto-Capture Protocol",
    "author": "AutoLab",
    "description": "Liquid handling with automatic camera capture after each action.",
}
requirements = {"robotType": "OT-2", "apiLevel": "2.18"}


# ── Runtime Parameters (editable in the Opentrons App before each run) ─────
def add_parameters(parameters: protocol_api.Parameters):
    parameters.add_str(
        variable_name="camera_server_ip",
        display_name="Camera Server IP",
        description="IP address of the computer running camera_server.py",
        default="172.26.4.16",
        choices=[
            {"display_name": "172.26.4.16", "value": "172.26.4.16"},
            {"display_name": "172.26.4.17", "value": "172.26.4.17"},
            {"display_name": "192.168.1.100", "value": "192.168.1.100"},
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
        variable_name="num_samples",
        display_name="Number of Samples",
        description="How many wells to process (starting from A1)",
        default=3,
        minimum=1,
        maximum=96,
    )
    parameters.add_int(
        variable_name="aspirate_volume",
        display_name="Aspirate Volume (uL)",
        default=10,
        minimum=1,
        maximum=20,
    )
    parameters.add_bool(
        variable_name="capture_enabled",
        display_name="Enable Camera Capture",
        description="Turn off to run without taking pictures",
        default=True,
    )
    parameters.add_bool(
        variable_name="dry_run",
        display_name="Dry Run",
        description="If true, skip actual liquid handling (for testing camera)",
        default=False,
    )


# ── Camera Capture Helper ──────────────────────────────────────────────────
def make_capture_fn(protocol: protocol_api.ProtocolContext):
    """
    Returns a function that sends a capture request to the camera server.
    Uses urllib (available in Python stdlib on the OT-2) to avoid
    needing to install requests.
    """

    base_url = (
        f"http://{protocol.params.camera_server_ip}"  # type: ignore[attr-defined]
        f":{protocol.params.camera_server_port}"       # type: ignore[attr-defined]
    )
    enabled = protocol.params.capture_enabled           # type: ignore[attr-defined]

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
    params = protocol.params  # type: ignore[attr-defined]

    # ── Load Labware ───────────────────────────────────────────────────
    tip_rack = protocol.load_labware("opentrons_96_tiprack_20ul", "1")

    # Camera position: load a dummy labware in slot 7 to move over it
    camera_spot = protocol.load_labware("corning_96_wellplate_360ul_flat", "7")

    # ── Load Pipette ───────────────────────────────────────────────────
    pipette = protocol.load_instrument(
        "p20_multi_gen2", mount="left", tip_racks=[tip_rack]
    )

    # ── Repeat pick up and return 5 times ────────────────────────────
    for i in range(5):
        protocol.comment(f"--- Round {i + 1}/5: Picking up tips from column 1 ---")
        pipette.pick_up_tip(tip_rack["A1"])

        # Move to slot 7 for camera, wait, then capture
        pipette.move_to(camera_spot["A1"].top(20))  # 20mm above well top
        protocol.delay(seconds=2)  # Let vibrations settle
        capture("pick_up_tip", f"round_{i+1}")
        protocol.delay(seconds=3)

        # Return tips to column 1
        protocol.comment(f"--- Round {i + 1}/5: Returning tips to column 1 ---")
        pipette.return_tip()

        # Move back to slot 7 for photo after drop
        pipette.move_to(camera_spot["A1"].top(20))
        protocol.delay(seconds=2)
        capture("return_tip", f"round_{i+1}")
        protocol.delay(seconds=3)

    protocol.comment("Protocol complete.")
