"""
Camera Server - Runs on your computer (not the OT-2).
Listens for HTTP requests and captures images from a USB camera using OpenCV.

Usage:
    python camera_server.py

The server listens on port 8080 by default.
Make sure the OT-2 and this computer are on the same network.

NOTE: On macOS, camera access must happen on the main thread.
      This server uses a threading architecture that keeps camera
      operations on the main thread while handling HTTP in a background thread.
"""

import cv2
import os
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────
CAMERA_INDEX = 1          # USB camera index (0 = first camera)
PORT = 8080               # Port the server listens on
SAVE_DIR = "./captured_images"  # Where images are saved
# ───────────────────────────────────────────────────────────────────────────

# Shared state between HTTP thread and main thread
capture_request = threading.Event()
capture_result = {"path": None, "error": None}
capture_done = threading.Event()
capture_params = {"action": "", "details": ""}
camera_lock = threading.Lock()


class CameraHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path == "/capture":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
            try:
                params = json.loads(body)
            except json.JSONDecodeError:
                params = {}

            action = params.get("action", "unknown")
            details = params.get("details", "")

            # Signal the main thread to capture
            with camera_lock:
                capture_params["action"] = action
                capture_params["details"] = details
                capture_done.clear()
                capture_request.set()

            # Wait for main thread to complete the capture
            capture_done.wait(timeout=10)

            if capture_result["path"]:
                response = {"status": "ok", "image_path": capture_result["path"]}
                self.send_response(200)
            else:
                response = {"status": "error", "message": capture_result.get("error", "Failed")}
                self.send_response(500)

            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        print(f"[SERVER] {args[0]}")


def capture_image(camera, action, details):
    """Capture an image — must be called from the main thread on macOS."""
    ret, frame = camera.read()
    if not ret:
        print(f"[ERROR] Failed to capture frame from camera {CAMERA_INDEX}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{action}.jpg"
    if details:
        safe_details = details.replace(" ", "_").replace("/", "-")
        filename = f"{timestamp}_{action}_{safe_details}.jpg"

    filepath = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(filepath, frame)
    print(f"[CAPTURE] {action}: {filepath}")
    return filepath


def run_server():
    """Run HTTP server in a background thread."""
    server = HTTPServer(("0.0.0.0", PORT), CameraHandler)
    server.serve_forever()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Open camera on main thread (required by macOS)
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print(f"[ERROR] Cannot open camera at index {CAMERA_INDEX}")
        print("  Try changing CAMERA_INDEX (0, 1, 2, ...)")
        return

    ret, _ = camera.read()
    if not ret:
        print(f"[ERROR] Camera opened but cannot read frames")
        camera.release()
        return
    print(f"[OK] Camera {CAMERA_INDEX} is working")

    # Start HTTP server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print(f"[OK] Camera server running on http://0.0.0.0:{PORT}")
    print(f"[OK] Images will be saved to {os.path.abspath(SAVE_DIR)}")
    print(f"[INFO] Press Ctrl+C to stop\n")

    # Main loop: wait for capture requests and handle them on main thread
    try:
        while True:
            # Wait for a capture request (check every 0.1s so Ctrl+C works)
            if capture_request.wait(timeout=0.1):
                capture_request.clear()

                with camera_lock:
                    action = capture_params["action"]
                    details = capture_params["details"]

                path = capture_image(camera, action, details)

                capture_result["path"] = path
                capture_result["error"] = None if path else "Capture failed"
                capture_done.set()

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
