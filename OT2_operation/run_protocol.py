"""
Universal OT-2 Protocol Runner
===============================
Give it any protocol .py file and it will:
  1. Read the protocol and extract its name, description, and runtime parameters
  2. Connect to the robot and check calibration status
  3. Prompt you to fill in (or accept defaults for) all runtime parameters
  4. Upload and run the protocol, streaming live status

Usage:
    python run_protocol.py                              # file picker
    python run_protocol.py path/to/my_protocol.py      # direct
    python run_protocol.py path/to/my_protocol.py --simulate   # dry-run (no robot movement)
"""

import ast
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── Robot connection ────────────────────────────────────────────────────────
ROBOT_IPS   = ["169.254.84.3", "172.26.4.16", "172.26.4.17"]
ROBOT_PORT  = 31950


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def _req(method, url, data=None, headers=None, timeout=15):
    if isinstance(data, dict):
        data = json.dumps(data).encode()
        ct = "application/json"
    else:
        ct = None
    req = urllib.request.Request(url, data=data, method=method)
    if ct:
        req.add_header("Content-Type", ct)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode(errors="replace")
            try:
                return r.status, json.loads(body)
            except json.JSONDecodeError:
                return r.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body
    except urllib.error.URLError as e:
        return None, str(e)

def _get(url, **kw):    return _req("GET",    url, **kw)
def _post(url, **kw):   return _req("POST",   url, **kw)
def _delete(url, **kw): return _req("DELETE", url, **kw)


# ── Protocol file parser ──────────────────────────────────────────────────────

def parse_protocol(path: Path) -> dict:
    """
    Parse a protocol .py file using AST (no execution).
    Returns:
        {
          "name": str,
          "description": str,
          "api_level": str,
          "robot_type": str,
          "params": [
              {
                "variable_name": str,
                "display_name": str,
                "description": str,
                "type": "int"|"float"|"str"|"bool",
                "default": ...,
                "minimum": ...,   # int/float only
                "maximum": ...,   # int/float only
                "choices": [...], # str only  [{display_name, value}]
              }, ...
          ]
        }
    """
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    info = {
        "name": path.stem,
        "description": "",
        "api_level": "?",
        "robot_type": "OT-2",
        "params": [],
    }

    # ── metadata dict ──────────────────────────────────────────────────
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "metadata":
                    if isinstance(node.value, ast.Dict):
                        for k, v in zip(node.value.keys, node.value.values):
                            key = ast.literal_eval(k) if isinstance(k, ast.Constant) else None
                            try:
                                val = ast.literal_eval(v)
                            except Exception:
                                val = None
                            if key == "protocolName" and val:
                                info["name"] = val
                            elif key == "description" and val:
                                info["description"] = val

        # ── requirements dict ────────────────────────────────────────
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "requirements":
                    if isinstance(node.value, ast.Dict):
                        for k, v in zip(node.value.keys, node.value.values):
                            key = ast.literal_eval(k) if isinstance(k, ast.Constant) else None
                            try:
                                val = ast.literal_eval(v)
                            except Exception:
                                val = None
                            if key == "apiLevel" and val:
                                info["api_level"] = val
                            elif key == "robotType" and val:
                                info["robot_type"] = val

    # ── add_parameters function ───────────────────────────────────────
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "add_parameters":
                for stmt in ast.walk(node):
                    if not isinstance(stmt, ast.Call):
                        continue
                    # Match parameters.add_int / add_str / add_float / add_bool
                    func = stmt.func
                    if not (isinstance(func, ast.Attribute) and
                            func.attr in ("add_int", "add_str", "add_float", "add_bool")):
                        continue

                    ptype = func.attr[4:]  # "int", "str", "float", "bool"

                    # Collect all keyword args
                    kw = {}
                    for knode in stmt.keywords:
                        try:
                            kw[knode.arg] = ast.literal_eval(knode.value)
                        except Exception:
                            kw[knode.arg] = None

                    # Handle choices list  [{display_name:..., value:...}, ...]
                    if "choices" in kw and isinstance(kw["choices"], list):
                        choices = []
                        for item in kw["choices"]:
                            if isinstance(item, dict):
                                choices.append(item)
                        kw["choices"] = choices

                    param = {
                        "variable_name": kw.get("variable_name", ""),
                        "display_name":  kw.get("display_name", kw.get("variable_name", "")),
                        "description":   kw.get("description", ""),
                        "type":          ptype,
                        "default":       kw.get("default"),
                    }
                    if ptype in ("int", "float"):
                        param["minimum"] = kw.get("minimum")
                        param["maximum"] = kw.get("maximum")
                    if ptype == "str" and "choices" in kw:
                        param["choices"] = kw["choices"]

                    info["params"].append(param)

    return info


# ── Calibration checker ───────────────────────────────────────────────────────

def check_calibration(base: str) -> bool:
    """
    Check pipette offset and deck calibration.
    Prints a summary and returns True if everything looks calibrated.
    """
    print("\n── Calibration Check ──────────────────────────────────")
    ok = True

    # Deck calibration
    status, deck = _get(f"{base}/calibration/status")
    if status == 200 and isinstance(deck, dict):
        deck_data = deck.get("deckCalibration", {})
        deck_status = deck_data.get("status", "unknown")
        marker = "OK" if deck_status == "OK" else "!!"
        print(f"  [{marker}] Deck calibration : {deck_status}")
        if deck_status != "OK":
            ok = False
    else:
        # Older API — try /robot/deck_calibration
        status2, deck2 = _get(f"{base}/robot/deck_calibration")
        if status2 == 200 and isinstance(deck2, dict):
            cal_type = deck2.get("deckCalibration", {}).get("type", "identity")
            marker = "OK" if cal_type != "identity" else "!!"
            print(f"  [{'OK' if cal_type != 'identity' else '!!'}] Deck calibration : {cal_type}")
            if cal_type == "identity":
                ok = False
        else:
            print("  [??] Deck calibration : could not retrieve")

    # Pipette offset calibrations
    status, pip_data = _get(f"{base}/calibration/pipette/offset")
    if status == 200 and isinstance(pip_data, dict):
        cals = pip_data.get("data", [])
        if cals:
            for cal in cals:
                mount  = cal.get("mount", "?")
                pip_id = cal.get("pipette", {}).get("id", "?")[:8]
                last   = cal.get("lastModified", "never")
                status_str = cal.get("status", {}).get("markedBad", False)
                marker = "!!" if status_str else "OK"
                print(f"  [{marker}] Pipette ({mount:<5}) : id={pip_id}...  calibrated={last[:10] if last != 'never' else 'NO'}")
                if status_str:
                    ok = False
        else:
            print("  [!!] No pipette offset calibrations found — run calibration first")
            ok = False
    else:
        # Try /pipettes
        status3, pips = _get(f"{base}/pipettes")
        if status3 == 200 and isinstance(pips, dict):
            for mount, pip in pips.items():
                if pip.get("model"):
                    cal_data = pip.get("calibration", None)
                    marker = "OK" if cal_data else "??"
                    print(f"  [{marker}] Pipette ({mount:<5}) : {pip.get('model','?')}  offset={'yes' if cal_data else 'unknown'}")
        else:
            print("  [??] Pipette calibration : could not retrieve")

    print("────────────────────────────────────────────────────────")
    return ok


# ── Parameter prompter ────────────────────────────────────────────────────────

def prompt_params(params: list) -> dict:
    """Interactively ask the user to confirm or override each runtime parameter."""
    if not params:
        return {}

    print("\n── Runtime Parameters ──────────────────────────────────")
    print("  Press Enter to accept the default, or type a new value.\n")

    values = {}
    for p in params:
        name    = p["display_name"] or p["variable_name"]
        ptype   = p["type"]
        default = p["default"]
        desc    = p.get("description", "")
        choices = p.get("choices", [])

        label = f"  {name}"
        if desc:
            label += f"\n    ({desc})"

        if choices:
            label += "\n    Choices:"
            for i, c in enumerate(choices):
                marker = " *" if c["value"] == default else "  "
                label += f"\n   {marker} {i+1}) {c['display_name']}  [{c['value']}]"
            label += f"\n    Enter number or value [{default}]: "
            raw = input(label).strip()
            if not raw:
                values[p["variable_name"]] = default
            elif raw.isdigit() and 1 <= int(raw) <= len(choices):
                values[p["variable_name"]] = choices[int(raw) - 1]["value"]
            else:
                values[p["variable_name"]] = raw

        elif ptype == "bool":
            cur = "Y" if default else "N"
            raw = input(f"{label} [Y/n, default={cur}]: ").strip().lower()
            if not raw:
                values[p["variable_name"]] = default
            else:
                values[p["variable_name"]] = raw in ("y", "yes", "1", "true")

        elif ptype in ("int", "float"):
            mn = p.get("minimum")
            mx = p.get("maximum")
            range_str = ""
            if mn is not None and mx is not None:
                range_str = f"  range {mn}–{mx}"
            while True:
                raw = input(f"{label}{range_str}  [default={default}]: ").strip()
                if not raw:
                    values[p["variable_name"]] = default
                    break
                try:
                    val = int(raw) if ptype == "int" else float(raw)
                    if mn is not None and val < mn:
                        print(f"    [!] Must be >= {mn}")
                        continue
                    if mx is not None and val > mx:
                        print(f"    [!] Must be <= {mx}")
                        continue
                    values[p["variable_name"]] = val
                    break
                except ValueError:
                    print(f"    [!] Please enter a valid {ptype}.")

        else:  # str
            raw = input(f"{label}  [default={default!r}]: ").strip()
            values[p["variable_name"]] = raw if raw else default

    print("────────────────────────────────────────────────────────")
    return values


# ── Robot helpers ─────────────────────────────────────────────────────────────

def find_robot(timeout=3) -> str:
    for ip in ROBOT_IPS:
        s, _ = _get(f"http://{ip}:{ROBOT_PORT}/health", timeout=timeout)
        if s == 200:
            return ip
    raise ConnectionError(
        f"No OT-2 found at: {ROBOT_IPS}\n"
        "  • Is the USB cable plugged in?\n"
        "  • Is the robot powered on?\n"
        "  • Check ROBOT_IPS at the top of this file."
    )

def upload_protocol(base, path: Path) -> str:
    file_bytes = path.read_bytes()
    boundary = "----OT2Boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files"; filename="{path.name}"\r\n'
        f"Content-Type: text/x-python\r\n\r\n"
    ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{base}/protocols", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        data = json.loads(e.read().decode())

    if not (isinstance(data, dict) and "data" in data):
        raise RuntimeError(f"Upload failed: {data}")

    pid = data["data"]["id"]

    # Check for analysis errors
    errors = data["data"].get("analyses", [])
    for a in data["data"].get("analysisSummaries", []):
        if a.get("status") not in ("completed", "pending"):
            print(f"  [!] Analysis warning: {a}")

    return pid

def create_and_start(base, protocol_id, rtp: dict) -> str:
    payload = {"data": {"protocolId": protocol_id}}
    if rtp:
        payload["data"]["runTimeParameterValues"] = rtp
    s, data = _post(f"{base}/runs", data=payload)
    if not (isinstance(data, dict) and "data" in data):
        raise RuntimeError(f"Could not create run (HTTP {s}): {data}")
    run_id = data["data"]["id"]
    _post(f"{base}/runs/{run_id}/actions", data={"data": {"actionType": "play"}})
    return run_id

def wait_for_run(base, run_id):
    terminal = {"succeeded", "failed", "stopped"}
    last = None
    print(f"\n── Live Status  (Ctrl-C to detach) ────────────────────")
    try:
        while True:
            _, data = _get(f"{base}/runs/{run_id}")
            run = data.get("data", {}) if isinstance(data, dict) else {}
            status = run.get("status", "unknown")
            if status != last:
                ts = time.strftime("%H:%M:%S")
                print(f"  [{ts}]  {status}")
                last = status
            if status in terminal:
                errors = run.get("errors", [])
                if errors:
                    print("\n  [!] Run errors:")
                    for e in errors:
                        print(f"      {e.get('errorType','?')}: {e.get('detail','')}")
                else:
                    print(f"\n  Run {status}.")
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n  Detached — run continues on robot.")
    print("────────────────────────────────────────────────────────\n")


# ── Simulation (dry-run via protocol analysis) ────────────────────────────────

def run_simulation(base, protocol_id: str) -> None:
    """
    Wait for the robot's protocol analysis to finish, then print every
    command that WOULD run — no motors move, no tips picked up.

    The OT-2 automatically analyses every uploaded protocol; analysis IS
    the simulation (it executes the protocol in a Python sandbox on the robot
    and records every command with its parameters).
    """
    print("\n── Simulation (dry-run) ────────────────────────────────")
    print("  Waiting for robot to analyse protocol...")

    # Get the analysis ID from the protocol record
    for attempt in range(30):          # wait up to ~60 s
        _, data = _get(f"{base}/protocols/{protocol_id}")
        if not isinstance(data, dict):
            time.sleep(2)
            continue
        summaries = data.get("data", {}).get("analysisSummaries", [])
        if summaries:
            latest = summaries[-1]
            analysis_id = latest["id"]
            status      = latest.get("status", "pending")
            if status == "completed":
                break
            if status == "failed":
                print(f"  [!] Analysis failed: {latest}")
                return
        time.sleep(2)
    else:
        print("  [!] Timed out waiting for analysis. Try again in a moment.")
        return

    # Fetch full analysis
    _, adata = _get(f"{base}/protocols/{protocol_id}/analyses/{analysis_id}")
    if not isinstance(adata, dict):
        print(f"  [!] Could not fetch analysis: {adata}")
        return

    analysis = adata.get("data", adata)

    # ── Errors / warnings ───────────────────────────────────────────────
    errors   = analysis.get("errors",   [])
    warnings = analysis.get("warnings", [])

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    [!!] {e.get('errorType','?')}: {e.get('detail','')}")
    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    [!]  {w.get('warningType','?')}: {w.get('detail','')}")
    if not errors and not warnings:
        print("  No errors or warnings.")

    # ── Command list ────────────────────────────────────────────────────
    commands = analysis.get("commands", [])
    print(f"\n  Commands ({len(commands)} total):\n")

    # Group commands for readability
    SHOW_DETAILS = {
        "pickUpTip", "dropTip", "returnTip",
        "aspirate", "dispense", "blowout",
        "moveToWell", "moveToCoordinates", "moveRelative",
        "loadLabware", "loadPipette", "loadLiquid",
        "waitForDuration", "waitForResume", "comment",
    }

    for i, cmd in enumerate(commands, 1):
        ctype  = cmd.get("commandType", "?")
        cstatus = cmd.get("status", "")
        params = cmd.get("params", {})
        result = cmd.get("result", {})

        # Build a short human-readable summary
        if ctype == "aspirate":
            well   = params.get("wellName", "?")
            labware = params.get("labwareId", "")[:8]
            vol    = params.get("volume", "?")
            detail = f"{vol} µL from {well} (labware …{labware})"
        elif ctype == "dispense":
            well   = params.get("wellName", "?")
            labware = params.get("labwareId", "")[:8]
            vol    = params.get("volume", "?")
            detail = f"{vol} µL into {well} (labware …{labware})"
        elif ctype == "pickUpTip":
            well   = params.get("wellName", "?")
            labware = params.get("labwareId", "")[:8]
            detail = f"from {well} (labware …{labware})"
        elif ctype in ("dropTip", "returnTip"):
            well   = params.get("wellName", "?")
            detail = f"at {well}" if well else ""
        elif ctype == "loadLabware":
            loc    = params.get("location", {})
            slot   = loc.get("slotName", loc.get("addressableAreaName", "?"))
            lname  = params.get("loadName", "?")
            detail = f"slot {slot}: {lname}"
        elif ctype == "loadPipette":
            mount  = params.get("mount", "?")
            pname  = params.get("pipetteName", "?")
            detail = f"{pname} on {mount}"
        elif ctype == "comment":
            detail = params.get("message", "")
        elif ctype == "waitForDuration":
            detail = f"{params.get('seconds', '?')} s"
        elif ctype == "blowout":
            well   = params.get("wellName", "?")
            detail = f"at {well}"
        else:
            detail = ""

        if ctype in SHOW_DETAILS or True:   # show all
            marker = "!!" if cstatus == "failed" else "  "
            print(f"  {marker}{i:>4}. {ctype:<28}  {detail}")

    # ── Liquid summary ──────────────────────────────────────────────────
    liquids = analysis.get("liquids", [])
    if liquids:
        print(f"\n  Liquids defined ({len(liquids)}):")
        for liq in liquids:
            print(f"    • {liq.get('displayName','?')}  {liq.get('description','')}")

    print("\n────────────────────────────────────────────────────────")
    print("  Simulation complete — no robot movement occurred.")
    print("────────────────────────────────────────────────────────\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Get protocol file + flags ────────────────────────────────────
    cli_args = sys.argv[1:]
    force_simulate = "--simulate" in cli_args
    cli_args = [a for a in cli_args if a != "--simulate"]

    if cli_args:
        protocol_path = Path(cli_args[0])
    else:
        print("OT-2 Universal Protocol Runner")
        print("=" * 40)
        raw = input("Protocol file path: ").strip()
        if not raw:
            print("No file given. Exiting.")
            sys.exit(0)
        protocol_path = Path(raw)

    if not protocol_path.exists():
        print(f"[ERROR] File not found: {protocol_path}")
        sys.exit(1)

    # ── 2. Parse the protocol ───────────────────────────────────────────
    print(f"\nParsing {protocol_path.name}...")
    proto = parse_protocol(protocol_path)

    print(f"\n  Protocol : {proto['name']}")
    if proto["description"]:
        print(f"  Desc     : {proto['description']}")
    print(f"  Robot    : {proto['robot_type']}  (API {proto['api_level']})")
    print(f"  Params   : {len(proto['params'])} runtime parameter(s)")

    # ── 3. Connect to robot ─────────────────────────────────────────────
    print("\nSearching for OT-2...")
    try:
        ip = find_robot()
    except ConnectionError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    base = f"http://{ip}:{ROBOT_PORT}"
    print(f"  Connected: {base}")

    # ── 4. Ask simulate or real run (skip if --simulate flag already set) ─
    if not force_simulate:
        print("\nHow would you like to run?")
        print("  1) Real run  — robot moves, tips picked up, liquid transferred")
        print("  2) Simulate  — dry-run only, no motors move (shows all commands)")
        mode = input("Choice [1/2, default=1]: ").strip()
        simulate = mode == "2"
    else:
        simulate = True

    if simulate:
        print("\n  [SIM] Simulation mode selected — robot will NOT move.")

    # ── 5. Calibration check (real runs only) ───────────────────────────
    if not simulate:
        cal_ok = check_calibration(base)
        if not cal_ok:
            print("\n  [!] Calibration issues detected.")
            print("      You can still run, but results may be inaccurate.")
            answer = input("  Continue anyway? (y/N): ").strip().lower()
            if answer != "y":
                print("Exiting. Please calibrate via the Opentrons App or robot touchscreen.")
                sys.exit(0)

    # ── 6. Runtime parameters ───────────────────────────────────────────
    rtp = prompt_params(proto["params"])

    # ── 7. Confirm ──────────────────────────────────────────────────────
    mode_label = "SIMULATE (dry-run)" if simulate else "REAL RUN"
    print(f"\n── Summary ─────────────────────────────────────────────")
    print(f"  Protocol : {proto['name']}")
    print(f"  Robot    : {ip}")
    print(f"  Mode     : {mode_label}")
    if rtp:
        print(f"  Parameters:")
        for k, v in rtp.items():
            print(f"    {k} = {v!r}")
    else:
        print(f"  Parameters: (none / all defaults)")

    action_label = "Simulate" if simulate else "Upload and run"
    answer = input(f"\n{action_label}? (Y/n): ").strip().lower()
    if answer == "n":
        print("Cancelled.")
        sys.exit(0)

    # ── 8. Upload protocol ───────────────────────────────────────────────
    print(f"\nUploading {protocol_path.name}...")
    try:
        protocol_id = upload_protocol(base, protocol_path)
        print(f"  Protocol ID: {protocol_id[:16]}...")
    except RuntimeError as e:
        print(f"[ERROR] Upload failed: {e}")
        sys.exit(1)

    # ── 9. Simulate or run ───────────────────────────────────────────────
    if simulate:
        run_simulation(base, protocol_id)
    else:
        print("Creating and starting run...")
        try:
            run_id = create_and_start(base, protocol_id, rtp)
            print(f"  Run ID: {run_id[:16]}...")
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        wait_for_run(base, run_id)


if __name__ == "__main__":
    main()
