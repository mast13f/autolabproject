"""
OT-2 Controller — HTTP API
==========================
Control the OT-2 robot directly without the Opentrons App.
Uses the robot's built-in REST API on port 31950.

Usage:
    python ot2_controller.py                          # interactive menu
    python ot2_controller.py upload my_protocol.py   # upload a protocol
    python ot2_controller.py run my_protocol.py      # upload + run
    python ot2_controller.py status                  # show robot status
    python ot2_controller.py stop                    # stop current run
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ── Default robot IPs (USB link-local is most reliable) ────────────────────
ROBOT_IPS = [
    "169.254.84.3",   # USB (link-local) — most common for direct connection
    "172.26.4.16",    # Wi-Fi / lab network
    "172.26.4.17",    # Wi-Fi / lab network (backup)
]
ROBOT_PORT = 31950


# ── HTTP helpers ────────────────────────────────────────────────────────────

def _request(method: str, url: str, data=None, content_type="application/json", timeout=10):
    """Send an HTTP request and return (status_code, parsed_json_or_None)."""
    if isinstance(data, dict):
        data = json.dumps(data).encode()
        content_type = "application/json"
    elif isinstance(data, bytes) and content_type != "multipart/form-data":
        pass  # already encoded

    req = urllib.request.Request(url, data=data, method=method)
    if data and not content_type.startswith("multipart"):
        req.add_header("Content-Type", content_type)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode(errors="replace")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body
    except urllib.error.URLError as e:
        return None, str(e)


def _get(url, **kw):
    return _request("GET", url, **kw)

def _post(url, data=None, **kw):
    return _request("POST", url, data=data, **kw)

def _delete(url, **kw):
    return _request("DELETE", url, **kw)


# ── Robot class ─────────────────────────────────────────────────────────────

class OT2:
    """Direct HTTP controller for an OT-2 robot."""

    def __init__(self, ip: str = None, port: int = ROBOT_PORT):
        self.ip = ip or self._find_robot()
        self.port = port
        self.base = f"http://{self.ip}:{self.port}"
        print(f"[OT2] Connected to {self.base}")

    # ── Discovery ────────────────────────────────────────────────────────

    @staticmethod
    def _find_robot(timeout: int = 3) -> str:
        """Try each known IP and return the first that responds."""
        for ip in ROBOT_IPS:
            url = f"http://{ip}:{ROBOT_PORT}/health"
            status, _ = _get(url, timeout=timeout)
            if status == 200:
                print(f"[OT2] Found robot at {ip}")
                return ip
        raise ConnectionError(
            f"No OT-2 found at any of: {ROBOT_IPS}\n"
            "Check:\n"
            "  • USB cable connected and robot is on\n"
            "  • Robot and computer on same Wi-Fi\n"
            "  • Correct IP in ROBOT_IPS list"
        )

    # ── Health / Status ──────────────────────────────────────────────────

    def health(self) -> dict:
        """Return robot health info."""
        _, data = _get(f"{self.base}/health")
        return data

    def status(self) -> None:
        """Print a summary of the robot's current state."""
        h = self.health()
        if not isinstance(h, dict):
            print(f"[STATUS] Could not reach robot: {h}")
            return
        print(f"\n{'='*50}")
        print(f"  Robot name : {h.get('name', '?')}")
        print(f"  API version: {h.get('api_version', '?')}")
        print(f"  FW version : {h.get('fw_version', '?')}")
        print(f"  System     : {h.get('system_version', '?')}")

        # Current run
        runs = self.list_runs()
        active = [r for r in runs if r.get("status") in ("running", "paused")]
        if active:
            r = active[0]
            print(f"\n  Active run : {r['id']}")
            print(f"  Status     : {r['status']}")
        else:
            print(f"\n  No active run.")
        print(f"{'='*50}\n")

    # ── Protocols ────────────────────────────────────────────────────────

    def list_protocols(self) -> list:
        """Return list of uploaded protocols."""
        _, data = _get(f"{self.base}/protocols")
        if isinstance(data, dict):
            return data.get("data", [])
        return []

    def upload_protocol(self, protocol_path: str) -> str:
        """Upload a .py protocol file. Returns the protocol ID."""
        path = Path(protocol_path)
        if not path.exists():
            raise FileNotFoundError(f"Protocol not found: {protocol_path}")

        file_bytes = path.read_bytes()
        filename = path.name

        # Build multipart/form-data manually (no external libraries needed)
        boundary = "----OT2Boundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
            f"Content-Type: text/x-python\r\n\r\n"
        ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()

        url = f"{self.base}/protocols"
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            data = json.loads(e.read().decode())

        if isinstance(data, dict) and "data" in data:
            protocol_id = data["data"]["id"]
            print(f"[UPLOAD] Protocol uploaded: {protocol_id}")
            print(f"         File: {filename}")

            # Show any analysis errors/warnings
            meta = data["data"].get("analysisSummaries", [])
            for s in meta:
                if s.get("status") != "completed":
                    print(f"  [!] Analysis: {s.get('status')} — {s.get('errors', '')}")
            return protocol_id
        else:
            raise RuntimeError(f"Upload failed: {data}")

    def delete_protocol(self, protocol_id: str) -> None:
        status, data = _delete(f"{self.base}/protocols/{protocol_id}")
        print(f"[DELETE] Protocol {protocol_id}: HTTP {status}")

    # ── Runs ─────────────────────────────────────────────────────────────

    def list_runs(self) -> list:
        _, data = _get(f"{self.base}/runs")
        if isinstance(data, dict):
            return data.get("data", [])
        return []

    def create_run(self, protocol_id: str, run_time_params: dict = None) -> str:
        """Create a run from a protocol ID. Returns run ID."""
        payload = {"data": {"protocolId": protocol_id}}
        if run_time_params:
            payload["data"]["runTimeParameterValues"] = run_time_params

        status, data = _post(f"{self.base}/runs", data=payload)
        if isinstance(data, dict) and "data" in data:
            run_id = data["data"]["id"]
            print(f"[RUN] Created run: {run_id}")
            return run_id
        raise RuntimeError(f"Failed to create run (HTTP {status}): {data}")

    def start_run(self, run_id: str) -> None:
        """Start (or resume) a run."""
        payload = {"data": {"actionType": "play"}}
        status, data = _post(f"{self.base}/runs/{run_id}/actions", data=payload)
        if status in (200, 201):
            print(f"[RUN] Started: {run_id}")
        else:
            print(f"[RUN] Start failed (HTTP {status}): {data}")

    def pause_run(self, run_id: str) -> None:
        payload = {"data": {"actionType": "pause"}}
        _post(f"{self.base}/runs/{run_id}/actions", data=payload)
        print(f"[RUN] Paused: {run_id}")

    def stop_run(self, run_id: str) -> None:
        payload = {"data": {"actionType": "stop"}}
        _post(f"{self.base}/runs/{run_id}/actions", data=payload)
        print(f"[RUN] Stopped: {run_id}")

    def get_run(self, run_id: str) -> dict:
        _, data = _get(f"{self.base}/runs/{run_id}")
        if isinstance(data, dict):
            return data.get("data", data)
        return {}

    def get_run_errors(self, run_id: str) -> list:
        run = self.get_run(run_id)
        return run.get("errors", [])

    def stop_active_run(self) -> None:
        """Stop whatever run is currently active."""
        runs = self.list_runs()
        for r in runs:
            if r.get("status") in ("running", "paused"):
                self.stop_run(r["id"])
                return
        print("[RUN] No active run to stop.")

    # ── High-level: upload + run ─────────────────────────────────────────

    def run_protocol(
        self,
        protocol_path: str,
        run_time_params: dict = None,
        wait: bool = True,
        poll_interval: int = 5,
    ) -> str:
        """Upload a protocol file, create a run, and start it.

        Args:
            protocol_path:   Path to the .py protocol file.
            run_time_params: Dict of runtime parameter overrides, e.g.
                             {"num_columns": 6, "transfer_volume": 15}
            wait:            If True, block and stream progress until done.
            poll_interval:   Seconds between status polls when wait=True.

        Returns:
            run_id
        """
        protocol_id = self.upload_protocol(protocol_path)
        run_id = self.create_run(protocol_id, run_time_params)
        self.start_run(run_id)

        if wait:
            self._wait_for_run(run_id, poll_interval)

        return run_id

    def _wait_for_run(self, run_id: str, poll_interval: int = 5) -> None:
        """Block until run finishes, printing status updates."""
        terminal_states = {"succeeded", "failed", "stopped"}
        last_status = None
        print(f"\n[WAIT] Monitoring run {run_id} (Ctrl-C to detach)...")
        try:
            while True:
                run = self.get_run(run_id)
                status = run.get("status", "unknown")
                if status != last_status:
                    ts = time.strftime("%H:%M:%S")
                    print(f"  [{ts}] Status: {status}")
                    last_status = status
                if status in terminal_states:
                    errors = self.get_run_errors(run_id)
                    if errors:
                        print(f"\n[!] Run errors:")
                        for e in errors:
                            print(f"    {e.get('errorType','?')}: {e.get('detail','?')}")
                    else:
                        print(f"\n[OK] Run {status}.")
                    break
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\n[WAIT] Detached — run continues on robot.")

    # ── Lights ───────────────────────────────────────────────────────────

    def lights(self, on: bool) -> None:
        payload = {"on": on}
        _post(f"{self.base}/robot/lights", data=payload)
        print(f"[LIGHTS] {'ON' if on else 'OFF'}")

    # ── Home ─────────────────────────────────────────────────────────────

    def home(self) -> None:
        """Home all axes."""
        _post(f"{self.base}/robot/home", data={"target": "robot"})
        print("[HOME] Homing robot...")


# ── CLI entry point ─────────────────────────────────────────────────────────

def _interactive_menu(robot: OT2):
    while True:
        print("\nOT-2 Control Menu")
        print("  1. Robot status")
        print("  2. Upload & run protocol")
        print("  3. List uploaded protocols")
        print("  4. List runs")
        print("  5. Stop active run")
        print("  6. Home robot")
        print("  7. Lights on/off")
        print("  q. Quit")
        choice = input("Choice: ").strip().lower()

        if choice == "1":
            robot.status()

        elif choice == "2":
            path = input("Protocol file path: ").strip()
            print("Runtime param overrides (leave blank for defaults):")
            rtp = {}
            while True:
                kv = input("  key=value (or Enter to continue): ").strip()
                if not kv:
                    break
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    # Try to parse as int/bool
                    if v.lower() in ("true", "false"):
                        rtp[k] = v.lower() == "true"
                    else:
                        try:
                            rtp[k] = int(v)
                        except ValueError:
                            rtp[k] = v
            robot.run_protocol(path, run_time_params=rtp or None)

        elif choice == "3":
            protocols = robot.list_protocols()
            if not protocols:
                print("No protocols uploaded.")
            for p in protocols:
                print(f"  {p['id'][:8]}...  {p.get('metadata', {}).get('protocolName', p.get('files', [{}])[0].get('name', '?'))}")

        elif choice == "4":
            runs = robot.list_runs()
            if not runs:
                print("No runs.")
            for r in runs:
                print(f"  {r['id'][:8]}...  status={r.get('status','?')}")

        elif choice == "5":
            robot.stop_active_run()

        elif choice == "6":
            robot.home()

        elif choice == "7":
            on = input("Lights on? (y/n): ").strip().lower() == "y"
            robot.lights(on)

        elif choice == "q":
            break


def main():
    args = sys.argv[1:]

    # Allow overriding the IP via env or first arg if it looks like an IP
    ip = None
    if args and args[0].replace(".", "").isdigit():
        ip = args.pop(0)

    try:
        robot = OT2(ip=ip)
    except ConnectionError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not args:
        _interactive_menu(robot)
        return

    cmd = args[0]

    if cmd == "status":
        robot.status()

    elif cmd == "upload" and len(args) >= 2:
        robot.upload_protocol(args[1])

    elif cmd == "run" and len(args) >= 2:
        # Parse any key=value pairs after the protocol path
        rtp = {}
        for kv in args[2:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                if v.lower() in ("true", "false"):
                    rtp[k] = v.lower() == "true"
                else:
                    try:
                        rtp[k] = int(v)
                    except ValueError:
                        rtp[k] = v
        robot.run_protocol(args[1], run_time_params=rtp or None)

    elif cmd == "stop":
        robot.stop_active_run()

    elif cmd == "home":
        robot.home()

    elif cmd == "lights" and len(args) >= 2:
        robot.lights(args[1].lower() in ("on", "1", "true", "yes"))

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
