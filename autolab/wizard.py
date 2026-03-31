"""
AutoLab Setup Wizard
=====================
A 5-step web UI that runs before the campaign. It checks connections,
collects configuration, shows deck layout, verifies calibration, then
hands off to the live campaign dashboard.

Usage (called automatically by experiment_runner.py):
    wizard = SetupWizard(port=9999, robot_ips=["169.254.84.3"], camera_ip="127.0.0.1")
    wizard.start()
    config = wizard.wait_for_start()   # blocks until user clicks "Start Campaign"
    wizard.stop()
    # → use config dict to configure run_campaign()
"""

import json
import os
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


# ── HTML ──────────────────────────────────────────────────────────────────────

_WIZARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoLab Setup Wizard</title>
<style>
  :root {
    --bg: #0f172a; --card: #1e293b; --border: #334155;
    --accent: #3b82f6; --accent2: #6366f1;
    --green: #22c55e; --red: #ef4444; --yellow: #eab308;
    --text: #f1f5f9; --muted: #94a3b8;
    --radius: 12px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }

  /* ── Layout ── */
  .page { max-width: 820px; margin: 0 auto; padding: 32px 20px 60px; }
  header { text-align: center; margin-bottom: 40px; }
  header h1 { font-size: 1.8rem; font-weight: 700; background: linear-gradient(90deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  header p { color: var(--muted); margin-top: 6px; font-size: 0.95rem; }

  /* ── Progress bar ── */
  .progress { display: flex; align-items: center; gap: 0; margin-bottom: 40px; }
  .step-dot { display: flex; flex-direction: column; align-items: center; flex: 1; }
  .dot-circle { width: 36px; height: 36px; border-radius: 50%; border: 2px solid var(--border);
    display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 700;
    transition: all .3s; background: var(--card); color: var(--muted); z-index: 1; }
  .dot-circle.active { border-color: var(--accent); color: var(--accent); background: rgba(59,130,246,.15); }
  .dot-circle.done { border-color: var(--green); background: rgba(34,197,94,.15); color: var(--green); }
  .dot-label { font-size: 0.72rem; color: var(--muted); margin-top: 6px; text-align: center; }
  .dot-label.active { color: var(--accent); }
  .connector { flex: 1; height: 2px; background: var(--border); margin-top: -20px; }
  .connector.done { background: var(--green); }

  /* ── Cards ── */
  .card { background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 28px; margin-bottom: 20px; }
  .card h2 { font-size: 1.05rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
  .card .subtitle { color: var(--text); font-size: 0.92rem; margin-bottom: 20px; }

  /* ── Panels (hidden/shown) ── */
  .panel { display: none; }
  .panel.active { display: block; }

  /* ── Hardware settings ── */
  .hw-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 18px; }
  .hw-field { display: flex; flex-direction: column; gap: 5px; }
  .hw-field label { font-size: 0.78rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; display: flex; align-items: center; gap: 5px; }
  .hw-field input { padding: 8px 11px; background: var(--bg); border: 1px solid var(--border); border-radius: 7px; color: var(--text); font-size: 0.9rem; font-family: monospace; transition: border-color .2s; }
  .hw-field input:focus { outline: none; border-color: var(--accent); }
  .hw-field input::placeholder { color: #475569; font-style: italic; }
  .hw-hint { font-size: 0.75rem; color: var(--muted); }

  /* ── Connection items ── */
  .conn-list { display: flex; flex-direction: column; gap: 12px; }
  .conn-item { display: flex; align-items: center; gap: 14px; padding: 14px 18px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; }
  .conn-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--border); flex-shrink: 0; }
  .conn-info { flex: 1; }
  .conn-name { font-weight: 600; font-size: 0.95rem; }
  .conn-addr { color: var(--muted); font-size: 0.82rem; margin-top: 2px; }
  .badge { padding: 3px 10px; border-radius: 99px; font-size: 0.78rem; font-weight: 600; }
  .badge.checking { background: rgba(234,179,8,.15); color: var(--yellow); }
  .badge.ok { background: rgba(34,197,94,.15); color: var(--green); }
  .badge.fail { background: rgba(239,68,68,.15); color: var(--red); }
  .badge.dry { background: rgba(99,102,241,.15); color: #a5b4fc; }

  /* ── Toggle ── */
  .toggle-row { display: flex; align-items: center; gap: 14px; padding: 14px 18px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 16px; }
  .toggle-label { flex: 1; }
  .toggle-label strong { display: block; font-size: 0.95rem; }
  .toggle-label span { color: var(--muted); font-size: 0.82rem; }
  .toggle { position: relative; width: 46px; height: 26px; flex-shrink: 0; }
  .toggle input { opacity: 0; width: 0; height: 0; }
  .tslider { position: absolute; inset: 0; background: var(--border); border-radius: 99px; cursor: pointer; transition: .3s; }
  .tslider:before { content: ''; position: absolute; width: 20px; height: 20px; left: 3px; top: 3px; background: white; border-radius: 50%; transition: .3s; }
  input:checked + .tslider { background: var(--accent); }
  input:checked + .tslider:before { transform: translateX(20px); }

  /* ── Factor rows ── */
  .factor-row { display: flex; align-items: center; gap: 14px; padding: 14px 18px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 10px; transition: border-color .2s; }
  .factor-row:hover { border-color: var(--accent); }
  .factor-name { font-weight: 600; font-size: 0.95rem; }
  .factor-range { color: var(--muted); font-size: 0.8rem; margin-top: 2px; }
  .factor-meta { flex: 1; }
  .fixed-input-wrap { display: flex; align-items: center; gap: 8px; }
  .fixed-input-wrap label { font-size: 0.82rem; color: var(--muted); white-space: nowrap; }
  .fixed-input-wrap input[type=number] { width: 90px; padding: 6px 8px; background: var(--card); border: 1px solid var(--border); border-radius: 6px; color: var(--text); font-size: 0.88rem; }
  .fixed-input-wrap select { padding: 6px 8px; background: var(--card); border: 1px solid var(--border); border-radius: 6px; color: var(--text); font-size: 0.88rem; }

  /* ── Initial experiments ── */
  .init-options { display: flex; gap: 12px; margin-bottom: 18px; }
  .init-card { flex: 1; padding: 16px; border: 2px solid var(--border); border-radius: 10px; cursor: pointer; transition: all .2s; }
  .init-card:hover { border-color: var(--accent); }
  .init-card.selected { border-color: var(--accent); background: rgba(59,130,246,.08); }
  .init-card h4 { font-size: 0.95rem; font-weight: 600; margin-bottom: 4px; }
  .init-card p { font-size: 0.82rem; color: var(--muted); }
  .custom-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 12px; }
  .custom-table th { text-align: left; padding: 8px 10px; color: var(--muted); font-weight: 600; border-bottom: 1px solid var(--border); }
  .custom-table td { padding: 6px 10px; }
  .custom-table input, .custom-table select { width: 100%; padding: 4px 6px; background: var(--bg); border: 1px solid var(--border); border-radius: 4px; color: var(--text); font-size: 0.85rem; }
  .add-row-btn { margin-top: 10px; padding: 6px 14px; background: transparent; border: 1px dashed var(--border); color: var(--muted); border-radius: 6px; cursor: pointer; font-size: 0.85rem; }
  .add-row-btn:hover { border-color: var(--accent); color: var(--accent); }

  /* ── Criteria (number inputs replacing sliders) ── */
  .criteria-row { display: flex; align-items: flex-start; gap: 16px; padding: 14px 0; border-bottom: 1px solid var(--border); }
  .criteria-row:last-child { border-bottom: none; }
  .criteria-label { width: 220px; flex-shrink: 0; padding-top: 4px; }
  .criteria-label .cl-name { font-size: 0.9rem; font-weight: 500; }
  .criteria-label .cl-hint { font-size: 0.78rem; color: var(--muted); margin-top: 3px; }
  .criteria-input { display: flex; align-items: center; gap: 10px; flex: 1; }
  .criteria-input input[type=number] { width: 100px; padding: 7px 10px; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; color: var(--text); font-size: 0.92rem; font-weight: 600; }
  .criteria-input input[type=number]:focus { outline: none; border-color: var(--accent); }
  .criteria-input input.invalid { border-color: var(--red); }
  .range-hint { font-size: 0.78rem; color: var(--muted); }
  .field-error { font-size: 0.78rem; color: var(--red); display: none; }
  .field-error.visible { display: inline; }

  /* ── Deck diagram ── */
  .deck-grid { display: grid; grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(4, 90px); gap: 10px; margin-top: 16px; }
  .slot { border: 2px solid var(--border); border-radius: 10px; padding: 10px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 4px; text-align: center; }
  .slot.active { border-color: var(--accent); background: rgba(59,130,246,.07); }
  .slot.camera-slot { border-color: #a855f7; background: rgba(168,85,247,.07); }
  .slot.empty { background: transparent; opacity: .3; }
  .slot.trash { border-color: #ef4444; background: rgba(239,68,68,.07); opacity: .7; }
  .slot .slot-num { font-size: 0.68rem; color: var(--muted); font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
  .slot .slot-indicator { width: 28px; height: 28px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: 700; letter-spacing: .03em; }
  .slot.active .slot-indicator { background: rgba(59,130,246,.2); color: var(--accent); border: 1px solid rgba(59,130,246,.4); }
  .slot.camera-slot .slot-indicator { background: rgba(168,85,247,.2); color: #c084fc; border: 1px solid rgba(168,85,247,.4); }
  .slot.empty .slot-indicator { background: var(--border); color: var(--muted); }
  .slot .slot-name { font-size: 0.82rem; font-weight: 600; }
  .slot .slot-desc { font-size: 0.68rem; color: var(--muted); font-family: monospace; word-break: break-all; }
  .deck-legend { display: flex; gap: 20px; margin-top: 14px; flex-wrap: wrap; }
  .leg-item { display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: var(--muted); }
  .leg-dot { width: 12px; height: 12px; border-radius: 3px; }

  /* ── Calibration ── */
  .cal-list { display: flex; flex-direction: column; gap: 10px; }
  .cal-item { display: flex; align-items: flex-start; gap: 14px; padding: 14px 18px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; }
  .cal-indicator { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: 6px; }
  .cal-info { flex: 1; }
  .cal-name { font-weight: 600; font-size: 0.92rem; }
  .cal-detail { color: var(--muted); font-size: 0.82rem; margin-top: 3px; }
  .cal-status { font-size: 0.82rem; font-weight: 600; white-space: nowrap; }

  /* ── Review summary ── */
  .review-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .review-item { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .review-item .r-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; font-weight: 600; margin-bottom: 6px; }
  .review-item .r-value { font-size: 0.95rem; font-weight: 600; }
  .review-item .r-sub { font-size: 0.82rem; color: var(--muted); margin-top: 3px; }
  .path-link { font-family: monospace; font-size: 0.82rem; color: var(--accent); cursor: pointer; text-decoration: underline; word-break: break-all; }
  .path-link:hover { color: #93c5fd; }

  /* ── Buttons ── */
  .btn-row { display: flex; gap: 12px; justify-content: flex-end; margin-top: 28px; }
  .btn { padding: 11px 26px; border-radius: 8px; border: none; cursor: pointer; font-size: 0.92rem; font-weight: 600; transition: all .2s; }
  .btn-ghost { background: transparent; border: 1px solid var(--border); color: var(--muted); }
  .btn-ghost:hover { border-color: var(--accent); color: var(--accent); }
  .btn-primary { background: var(--accent); color: white; }
  .btn-primary:hover { background: #2563eb; }
  .btn-primary:disabled { background: var(--border); color: var(--muted); cursor: not-allowed; }
  .btn-green { background: var(--green); color: #052e16; }
  .btn-green:hover { background: #16a34a; }
  .btn-green:disabled { background: var(--border); color: var(--muted); cursor: not-allowed; }

  /* ── Tooltips ── */
  .tip { display: inline-flex; align-items: center; justify-content: center; width: 15px; height: 15px;
    border-radius: 50%; background: var(--border); color: var(--muted); font-size: 0.65rem; font-weight: 700;
    cursor: help; position: relative; flex-shrink: 0; vertical-align: middle; margin-left: 5px; }
  .tip::after { content: attr(data-tip); position: absolute; bottom: calc(100% + 8px); left: 50%; transform: translateX(-50%);
    white-space: normal; min-width: 200px; max-width: 300px; background: #1e293b; border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 13px; font-size: 0.8rem; color: var(--text); z-index: 200;
    pointer-events: none; line-height: 1.5; opacity: 0; transition: opacity .15s; font-weight: 400;
    box-shadow: 0 4px 20px rgba(0,0,0,.4); }
  .tip::before { content: ''; position: absolute; bottom: calc(100% + 3px); left: 50%; transform: translateX(-50%);
    border: 5px solid transparent; border-top-color: var(--border); z-index: 201; pointer-events: none;
    opacity: 0; transition: opacity .15s; }
  .tip:hover::after, .tip:hover::before { opacity: 1; }

  /* ── Misc ── */
  .note { background: rgba(59,130,246,.08); border: 1px solid rgba(59,130,246,.25); border-radius: 8px; padding: 12px 16px; font-size: 0.85rem; color: #93c5fd; margin-top: 14px; }
  .warn { background: rgba(234,179,8,.08); border: 1px solid rgba(234,179,8,.25); border-radius: 8px; padding: 12px 16px; font-size: 0.85rem; color: #fde68a; margin-top: 14px; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(255,255,255,.3); border-top-color: white; border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .section-label { font-size: 0.72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; display: flex; align-items: center; gap: 6px; }
</style>
</head>
<body>
<div class="page">

  <!-- Header -->
  <header>
    <h1>AutoLab Setup Wizard</h1>
    <p id="dry-run-badge" style="display:none">
      <span class="badge dry" style="font-size:0.88rem;padding:5px 16px">DRY RUN MODE — no robot required</span>
    </p>
  </header>

  <!-- Progress -->
  <div class="progress">
    <div class="step-dot">
      <div class="dot-circle active" id="dot-1">1</div>
      <div class="dot-label active" id="lbl-1">Connection</div>
    </div>
    <div class="connector" id="con-1"></div>
    <div class="step-dot">
      <div class="dot-circle" id="dot-2">2</div>
      <div class="dot-label" id="lbl-2">Configure</div>
    </div>
    <div class="connector" id="con-2"></div>
    <div class="step-dot">
      <div class="dot-circle" id="dot-3">3</div>
      <div class="dot-label" id="lbl-3">Deck Layout</div>
    </div>
    <div class="connector" id="con-3"></div>
    <div class="step-dot">
      <div class="dot-circle" id="dot-4">4</div>
      <div class="dot-label" id="lbl-4">Calibration</div>
    </div>
    <div class="connector" id="con-4"></div>
    <div class="step-dot">
      <div class="dot-circle" id="dot-5">5</div>
      <div class="dot-label" id="lbl-5">Review & Start</div>
    </div>
  </div>

  <!-- ═════════ STEP 1: CONNECTION ═════════ -->
  <div class="panel active" id="panel-1">
    <div class="card">
      <h2>Connection Check</h2>
      <div class="subtitle">Configure your hardware addresses, then verify everything is reachable.</div>

      <!-- Dry run toggle -->
      <div class="toggle-row">
        <div class="toggle-label">
          <strong>Dry Run Mode
            <span class="tip" data-tip="When enabled, the robot and camera are not used. The optimizer runs normally but accuracy scores are simulated with random values. Use this to test the full pipeline without hardware.">?</span>
          </strong>
          <span>Skip real hardware — simulate accuracy scores for pipeline testing</span>
        </div>
        <label class="toggle" title="Toggle dry run mode">
          <input type="checkbox" id="dry-run-toggle">
          <span class="tslider"></span>
        </label>
      </div>

      <!-- Hardware address settings -->
      <div class="section-label" style="margin-top:4px">
        Hardware Addresses
        <span class="tip" data-tip="Edit these fields to match your network setup. Changes take effect when you click Check Connections. The robot IP is tried first; leave it blank to auto-detect across common addresses.">?</span>
      </div>
      <div class="hw-grid">
        <div class="hw-field">
          <label>OT-2 Robot IP
            <span class="tip" data-tip="The IP address of your OT-2. Over USB this is usually 169.254.84.3. Leave blank to auto-detect from known addresses.">?</span>
          </label>
          <input type="text" id="robot-ip-input" placeholder="Auto-detect"
                 title="Robot IP address (leave blank to auto-detect)">
          <span class="hw-hint">Leave blank to auto-detect</span>
        </div>
        <div class="hw-field">
          <label>Camera Server IP
            <span class="tip" data-tip="The IP of the computer running camera_server.py. If the camera server is on the same machine connected to the OT-2 via USB, this is usually the same as the robot IP host.">?</span>
          </label>
          <input type="text" id="camera-ip-input" value="__CAMERA_IP__"
                 title="IP address of the computer running camera_server.py">
        </div>
        <div class="hw-field">
          <label>Camera Server Port
            <span class="tip" data-tip="The port camera_server.py is listening on. Default is 8080. Change this if you started the server on a different port.">?</span>
          </label>
          <input type="number" id="camera-port-input" value="__CAMERA_PORT__" min="1024" max="65535"
                 title="Port camera_server.py listens on (default 8080)">
        </div>
        <div class="hw-field">
          <label>Camera Device Index
            <span class="tip" data-tip="The OS index of the USB camera connected to the computer running camera_server.py. 0 is usually the built-in webcam, 1 is the first external USB camera. Change this if the wrong camera is being used.">?</span>
          </label>
          <input type="number" id="camera-index-input" value="__CAMERA_INDEX__" min="0" max="10"
                 title="USB camera device index (0 = built-in, 1 = first USB camera, etc.)">
          <span class="hw-hint">0 = built-in, 1 = first USB camera</span>
        </div>
      </div>

      <!-- Connection status -->
      <div class="section-label">Connection Status</div>
      <div class="conn-list">
        <div class="conn-item">
          <div class="conn-dot" id="robot-dot"></div>
          <div class="conn-info">
            <div class="conn-name">OT-2 Robot</div>
            <div class="conn-addr" id="robot-addr">Press Check Connections to start</div>
          </div>
          <span class="badge checking" id="robot-badge">—</span>
        </div>
        <div class="conn-item">
          <div class="conn-dot" id="camera-dot"></div>
          <div class="conn-info">
            <div class="conn-name">Camera Server</div>
            <div class="conn-addr" id="camera-addr">Press Check Connections to start</div>
          </div>
          <span class="badge checking" id="camera-badge">—</span>
        </div>
      </div>

      <div id="conn-note" class="note" style="display:none"></div>
    </div>

    <div class="btn-row">
      <button class="btn btn-ghost" onclick="checkConnections()" title="Test the connection to the robot and camera server">Check Connections</button>
      <button class="btn btn-primary" id="next-1" onclick="goStep(2)" disabled title="Proceed to experiment settings">Next</button>
    </div>
  </div>

  <!-- ═════════ STEP 2: CONFIGURE ═════════ -->
  <div class="panel" id="panel-2">
    <div class="card">
      <h2>Experiment Settings</h2>
      <div class="subtitle">Select which parameters to optimize and configure the run.</div>

      <div class="toggle-row">
        <div class="toggle-label">
          <strong>OT-2 Deck Lights
            <span class="tip" data-tip="Turns on the LED lights on the OT-2 deck during the run. Recommended for camera-based image capture to ensure consistent illumination.">?</span>
          </strong>
          <span>Illuminate the robot deck during the experiment</span>
        </div>
        <label class="toggle" title="Toggle OT-2 deck lights on/off">
          <input type="checkbox" id="lights-toggle" checked>
          <span class="tslider"></span>
        </label>
      </div>

      <div class="section-label">
        Parameters to Optimize
        <span class="tip" data-tip="Checked parameters are varied by the Bayesian optimizer each iteration. Unchecked parameters are held constant at the value you specify — useful when you already know a good value for that parameter.">?</span>
      </div>
      <div id="factor-list"><!-- Injected by JS --></div>
    </div>

    <div class="card">
      <h2>Initial Experiments
        <span class="tip" data-tip="Before Bayesian optimization begins, a set of initial experiments is run to build the surrogate model. These can be generated automatically or set manually.">?</span>
      </h2>
      <div class="subtitle">How should the first few experiments be selected?</div>

      <div class="init-options">
        <div class="init-card selected" id="init-algo" onclick="selectInit('algorithm')"
             title="Let the algorithm choose initial points using Latin Hypercube Sampling">
          <h4>Algorithm (LHS)
            <span class="tip" data-tip="Latin Hypercube Sampling (LHS) divides each parameter range into equal intervals and picks one value per interval, ensuring even coverage across the full parameter space.">?</span>
          </h4>
          <p>Automatically spread initial points across the parameter space.</p>
        </div>
        <div class="init-card" id="init-custom" onclick="selectInit('custom')"
             title="Enter specific parameter values for the first experiments">
          <h4>Custom</h4>
          <p>Manually specify the first experiment values — useful if you have known good settings to start from.</p>
        </div>
      </div>
      <div id="custom-table-wrap" style="display:none">
        <table class="custom-table">
          <thead><tr id="custom-header"></tr></thead>
          <tbody id="custom-body"></tbody>
        </table>
        <button class="add-row-btn" onclick="addCustomRow()" title="Add another experiment row">+ Add experiment</button>
      </div>
    </div>

    <div class="card">
      <h2>Stopping Criteria
        <span class="tip" data-tip="The campaign stops automatically when any criterion is met: the maximum number of experiments is reached, or the best result stops improving (convergence).">?</span>
      </h2>
      <div class="subtitle">The campaign ends when any of these conditions is reached.</div>

      <div class="criteria-row">
        <div class="criteria-label">
          <div class="cl-name">Max Experiments
            <span class="tip" data-tip="Hard cap on the total number of experiments, including both the initial phase and Bayesian optimization rounds. The campaign always stops at this number even if convergence has not been reached.">?</span>
          </div>
          <div class="cl-hint">Includes initial + optimization rounds</div>
        </div>
        <div class="criteria-input">
          <input type="number" id="max-iter" value="20" min="5" max="200" step="1"
                 oninput="validateInt('max-iter','err-max-iter',5,200)"
                 title="Total number of experiments to run (5–200)">
          <span class="range-hint">5 – 200</span>
          <span class="field-error" id="err-max-iter"></span>
        </div>
      </div>

      <div class="criteria-row">
        <div class="criteria-label">
          <div class="cl-name">Initial Points (LHS)
            <span class="tip" data-tip="Number of initial experiments run before Bayesian optimization starts. More points give the optimizer a better starting model but cost more experiments. Must be less than Max Experiments.">?</span>
          </div>
          <div class="cl-hint">Must be less than Max Experiments</div>
        </div>
        <div class="criteria-input">
          <input type="number" id="n-initial" value="5" min="3" max="30" step="1"
                 oninput="validateNInitial()"
                 title="Number of initial LHS experiments (3–30)">
          <span class="range-hint">3 – 30</span>
          <span class="field-error" id="err-n-initial"></span>
        </div>
      </div>

      <div class="criteria-row">
        <div class="criteria-label">
          <div class="cl-name">Convergence Window
            <span class="tip" data-tip="Number of consecutive Bayesian optimization iterations to evaluate for convergence. If the best result does not improve by more than the tolerance over this many iterations, the campaign is considered converged and stops early.">?</span>
          </div>
          <div class="cl-hint">Consecutive BO iterations to check</div>
        </div>
        <div class="criteria-input">
          <input type="number" id="conv-window" value="5" min="3" max="20" step="1"
                 oninput="validateInt('conv-window','err-conv-window',3,20)"
                 title="Convergence check window size (3–20 iterations)">
          <span class="range-hint">3 – 20</span>
          <span class="field-error" id="err-conv-window"></span>
        </div>
      </div>

      <div class="criteria-row">
        <div class="criteria-label">
          <div class="cl-name">Convergence Tolerance (%)
            <span class="tip" data-tip="Minimum improvement in accuracy (percentage points) required over the convergence window to be considered 'not converged'. If improvement falls below this value, the campaign stops early. Lower values are stricter.">?</span>
          </div>
          <div class="cl-hint">Min improvement to continue optimizing</div>
        </div>
        <div class="criteria-input">
          <input type="number" id="conv-tol" value="1.0" min="0.01" max="20" step="0.1"
                 oninput="validateFloat('conv-tol','err-conv-tol',0.01,20)"
                 title="Minimum accuracy improvement to continue (0.01–20%)">
          <span class="range-hint">0.01 – 20 %</span>
          <span class="field-error" id="err-conv-tol"></span>
        </div>
      </div>
    </div>

    <div class="btn-row">
      <button class="btn btn-ghost" onclick="goStep(1)" title="Go back to connection check">Back</button>
      <button class="btn btn-primary" onclick="tryGoStep3()" title="Proceed to deck layout">Next</button>
    </div>
  </div>

  <!-- ═════════ STEP 3: DECK LAYOUT ═════════ -->
  <div class="panel" id="panel-3">
    <div class="card">
      <h2>Deck Layout</h2>
      <div class="subtitle">Place labware in the correct slots on the OT-2 deck before starting. The diagram below shows the robot as viewed from the front.</div>

      <div class="deck-grid">
        <!-- Row 4 (top — back of robot): 10, 11, TRASH -->
        <div class="slot empty" title="Slot 10 — Not used in this protocol">
          <div class="slot-num">Slot 10</div>
          <div class="slot-indicator">—</div>
          <div class="slot-name">Empty</div>
        </div>
        <div class="slot empty" title="Slot 11 — Not used in this protocol">
          <div class="slot-num">Slot 11</div>
          <div class="slot-indicator">—</div>
          <div class="slot-name">Empty</div>
        </div>
        <div class="slot trash" title="Fixed trash bin — always top-right on OT-2">
          <div class="slot-num">TRASH</div>
          <div class="slot-indicator">🗑</div>
          <div class="slot-name">Trash Bin</div>
        </div>
        <!-- Row 3: 7, 8, 9 -->
        <div class="slot camera-slot" title="Slot 7 — Camera reservoir. The pipette moves here and the camera films the tips from above.">
          <div class="slot-num">Slot 7</div>
          <div class="slot-indicator">CAM</div>
          <div class="slot-name">Camera Reservoir</div>
          <div class="slot-desc">agilent_1_reservoir_290ml</div>
        </div>
        <div class="slot empty" title="Slot 8 — Not used in this protocol">
          <div class="slot-num">Slot 8</div>
          <div class="slot-indicator">—</div>
          <div class="slot-name">Empty</div>
        </div>
        <div class="slot empty" title="Slot 9 — Not used in this protocol">
          <div class="slot-num">Slot 9</div>
          <div class="slot-indicator">—</div>
          <div class="slot-name">Empty</div>
        </div>
        <!-- Row 2: 4, 5, 6 -->
        <div class="slot active" title="Slot 4 — Source reservoir. Fill with your sample liquid before starting.">
          <div class="slot-num">Slot 4</div>
          <div class="slot-indicator">SRC</div>
          <div class="slot-name">Source Reservoir</div>
          <div class="slot-desc">agilent_1_reservoir_290ml</div>
        </div>
        <div class="slot empty" title="Slot 5 — Not used in this protocol">
          <div class="slot-num">Slot 5</div>
          <div class="slot-indicator">—</div>
          <div class="slot-name">Empty</div>
        </div>
        <div class="slot empty" title="Slot 6 — Not used in this protocol">
          <div class="slot-num">Slot 6</div>
          <div class="slot-indicator">—</div>
          <div class="slot-name">Empty</div>
        </div>
        <!-- Row 1 (front, closest to user): 1, 2, 3 -->
        <div class="slot active" title="Slot 1 — 20 µL filter tip rack. Each transfer uses one column of tips.">
          <div class="slot-num">Slot 1</div>
          <div class="slot-indicator">TIP</div>
          <div class="slot-name">Tip Rack</div>
          <div class="slot-desc">opentrons_96_filtertiprack_20ul</div>
        </div>
        <div class="slot active" title="Slot 2 — 96-well plate. Liquid is dispensed into columns 1–12.">
          <div class="slot-num">Slot 2</div>
          <div class="slot-indicator">PLT</div>
          <div class="slot-name">Well Plate</div>
          <div class="slot-desc">corning_96_wellplate_330ul</div>
        </div>
        <div class="slot empty" title="Slot 3 — Not used in this protocol">
          <div class="slot-num">Slot 3</div>
          <div class="slot-indicator">—</div>
          <div class="slot-name">Empty</div>
        </div>
      </div>

      <div class="deck-legend">
        <div class="leg-item"><div class="leg-dot" style="background:rgba(59,130,246,.4);border:1.5px solid var(--accent)"></div> Required labware</div>
        <div class="leg-item"><div class="leg-dot" style="background:rgba(168,85,247,.4);border:1.5px solid #a855f7"></div> Camera position (Slot 7)</div>
        <div class="leg-item"><div class="leg-dot" style="background:rgba(239,68,68,.4);border:1.5px solid #ef4444"></div> Fixed trash bin</div>
        <div class="leg-item"><div class="leg-dot" style="background:var(--border)"></div> Empty — leave clear</div>
      </div>

      <div class="note">
        The multi-channel pipette is mounted on the <strong>left arm</strong>.
        Fill the <strong>source reservoir (Slot 4)</strong> with your liquid before clicking Next.
        Slot 7 acts as the camera filming position — leave it empty or place a labware-height placeholder.
      </div>
    </div>

    <div class="btn-row">
      <button class="btn btn-ghost" onclick="goStep(2)" title="Go back to experiment settings">Back</button>
      <button class="btn btn-primary" onclick="goStep(4)" title="Proceed to calibration check">Next</button>
    </div>
  </div>

  <!-- ═════════ STEP 4: CALIBRATION ═════════ -->
  <div class="panel" id="panel-4">
    <div class="card">
      <h2>Calibration Check</h2>
      <div class="subtitle">Verifying deck and pipette calibration via the OT-2 REST API. If calibration is missing, re-run it in the Opentrons app before proceeding.</div>

      <div class="cal-list" id="cal-list">
        <div class="cal-item">
          <div class="cal-indicator" style="background:var(--yellow)"></div>
          <div class="cal-info"><div class="cal-name">Loading calibration status…</div></div>
        </div>
      </div>

      <div id="cal-note" class="note" style="display:none"></div>
      <div id="cal-warn" class="warn" style="display:none"></div>
    </div>

    <div class="btn-row">
      <button class="btn btn-ghost" onclick="goStep(3)" title="Go back to deck layout">Back</button>
      <button class="btn btn-ghost" onclick="loadCalibration()" title="Re-query the robot for calibration status">Re-check</button>
      <button class="btn btn-primary" id="next-4" onclick="goStep(5)" title="Proceed to review">Next</button>
    </div>
  </div>

  <!-- ═════════ STEP 5: REVIEW & START ═════════ -->
  <div class="panel" id="panel-5">
    <div class="card">
      <h2>Review & Launch</h2>
      <div class="subtitle">Confirm all settings before starting the campaign. Click a file path to open its location.</div>

      <div class="review-grid" id="review-grid"><!-- Filled by JS --></div>
    </div>

    <div id="launch-note" class="note">
      Once launched, the campaign dashboard will open automatically.
      You can pause or stop the campaign from there at any time.
    </div>

    <div class="btn-row">
      <button class="btn btn-ghost" onclick="goStep(4)" title="Go back to calibration check">Back</button>
      <button class="btn btn-green" id="start-btn" onclick="launchCampaign()" title="Start the optimization campaign">Start Campaign</button>
    </div>
  </div>

</div><!-- /page -->

<script>
// ── State ────────────────────────────────────────────────────────────────────
let dryRun = false;
let lights = true;
let initMode = 'algorithm';
let currentStep = 1;
let connStatus = { robot: false, camera: false };
let calData = null;

const FACTORS_META = __FACTORS_META__;   // injected by Python

// ── Utilities ────────────────────────────────────────────────────────────────
function goStep(n) {
  document.getElementById(`panel-${currentStep}`).classList.remove('active');
  const oldDot = document.getElementById(`dot-${currentStep}`);
  oldDot.classList.remove('active'); oldDot.classList.add('done');
  document.getElementById(`lbl-${currentStep}`).classList.remove('active');
  if (currentStep < 5) document.getElementById(`con-${currentStep}`).classList.add('done');

  currentStep = n;
  document.getElementById(`panel-${n}`).classList.add('active');
  const dot = document.getElementById(`dot-${n}`);
  dot.classList.remove('done'); dot.classList.add('active');
  document.getElementById(`lbl-${n}`).classList.add('active');

  if (n === 4) loadCalibration();
  if (n === 5) buildReview();
}

// ── Input validation ─────────────────────────────────────────────────────────
function validateInt(inputId, errId, min, max) {
  const el = document.getElementById(inputId);
  const err = document.getElementById(errId);
  const v = parseInt(el.value);
  if (isNaN(v) || v < min || v > max || el.value.trim() === '') {
    el.classList.add('invalid');
    err.textContent = `Enter a whole number between ${min} and ${max}`;
    err.classList.add('visible');
    return false;
  }
  el.classList.remove('invalid');
  err.classList.remove('visible');
  return true;
}

function validateFloat(inputId, errId, min, max) {
  const el = document.getElementById(inputId);
  const err = document.getElementById(errId);
  const v = parseFloat(el.value);
  if (isNaN(v) || v < min || v > max || el.value.trim() === '') {
    el.classList.add('invalid');
    err.textContent = `Enter a number between ${min} and ${max}`;
    err.classList.add('visible');
    return false;
  }
  el.classList.remove('invalid');
  err.classList.remove('visible');
  return true;
}

function validateNInitial() {
  const ok1 = validateInt('n-initial', 'err-n-initial', 3, 30);
  if (!ok1) return false;
  const nInit = parseInt(document.getElementById('n-initial').value);
  const maxIter = parseInt(document.getElementById('max-iter').value);
  const err = document.getElementById('err-n-initial');
  if (!isNaN(maxIter) && nInit >= maxIter) {
    document.getElementById('n-initial').classList.add('invalid');
    err.textContent = `Must be less than Max Experiments (${maxIter})`;
    err.classList.add('visible');
    return false;
  }
  err.classList.remove('visible');
  document.getElementById('n-initial').classList.remove('invalid');
  return true;
}

function validateFixedInput(name, min, max) {
  const el = document.getElementById(`fixed-val-${name}`);
  const err = document.getElementById(`fixed-err-${name}`);
  if (!el || !err || el.tagName === 'SELECT') return true;
  const v = parseFloat(el.value);
  if (isNaN(v) || v < min || v > max) {
    el.style.borderColor = 'var(--red)';
    err.textContent = `${min} – ${max}`;
    err.style.display = 'inline';
    return false;
  }
  el.style.borderColor = '';
  err.style.display = 'none';
  return true;
}

function validateAllStep2() {
  let ok = true;
  ok = validateInt('max-iter', 'err-max-iter', 5, 200) && ok;
  ok = validateNInitial() && ok;
  ok = validateInt('conv-window', 'err-conv-window', 3, 20) && ok;
  ok = validateFloat('conv-tol', 'err-conv-tol', 0.01, 20) && ok;
  // Validate fixed-value inputs for unchecked factors
  FACTORS_META.forEach(f => {
    const cb = document.getElementById(`opt-${f.name}`);
    if (cb && !cb.checked && f.kind !== 'categorical') {
      ok = validateFixedInput(f.name, f.low, f.high) && ok;
    }
  });
  return ok;
}

function tryGoStep3() {
  if (validateAllStep2()) goStep(3);
}

// ── Step 1: Connections ───────────────────────────────────────────────────────
document.getElementById('dry-run-toggle').addEventListener('change', function() {
  dryRun = this.checked;
  document.getElementById('dry-run-badge').style.display = dryRun ? 'block' : 'none';
  if (dryRun) {
    setBadge('robot', 'dry', 'Dry Run');
    setBadge('camera', 'dry', 'Dry Run');
    setDot('robot-dot', '#6366f1');
    setDot('camera-dot', '#6366f1');
    document.getElementById('robot-addr').textContent = 'Skipped in dry run mode';
    document.getElementById('camera-addr').textContent = 'Skipped in dry run mode';
    document.getElementById('conn-note').style.display = 'block';
    document.getElementById('conn-note').textContent = 'Dry run enabled — hardware connections are not required.';
    document.getElementById('next-1').disabled = false;
  } else {
    checkConnections();
  }
});

function setBadge(which, cls, text) {
  const el = document.getElementById(`${which}-badge`);
  el.className = `badge ${cls}`;
  el.textContent = text;
}

function setDot(id, color) {
  document.getElementById(id).style.background = color;
}

async function checkConnections() {
  if (dryRun) return;
  setBadge('robot', 'checking', 'Checking…');
  setBadge('camera', 'checking', 'Checking…');
  setDot('robot-dot', 'var(--yellow)');
  setDot('camera-dot', 'var(--yellow)');
  const robotIpOverride = document.getElementById('robot-ip-input').value.trim();
  const cameraIp  = document.getElementById('camera-ip-input').value.trim();
  const cameraPort = parseInt(document.getElementById('camera-port-input').value) || 8080;
  const cameraIndex = parseInt(document.getElementById('camera-index-input').value) || 0;

  document.getElementById('robot-addr').textContent = robotIpOverride
    ? `Checking ${robotIpOverride}…`
    : 'Searching known IP addresses…';
  document.getElementById('camera-addr').textContent = '—';
  document.getElementById('next-1').disabled = true;
  document.getElementById('conn-note').style.display = 'none';

  try {
    const res = await fetch('/api/check-connections', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ robot_ip: robotIpOverride, camera_ip: cameraIp, camera_port: cameraPort, camera_index: cameraIndex }),
    });
    if (dryRun) return;
    const data = await res.json();

    if (data.robot.ok) {
      setBadge('robot', 'ok', 'Connected');
      setDot('robot-dot', 'var(--green)');
      document.getElementById('robot-addr').textContent = data.robot.ip || 'found';
      connStatus.robot = true;
    } else {
      setBadge('robot', 'fail', 'Not found');
      setDot('robot-dot', 'var(--red)');
      document.getElementById('robot-addr').textContent = 'Could not reach robot on any known IP';
      connStatus.robot = false;
    }

    if (data.camera.ok) {
      setBadge('camera', 'ok', 'Connected');
      setDot('camera-dot', 'var(--green)');
      document.getElementById('camera-addr').textContent = data.camera.addr;
      connStatus.camera = true;
    } else {
      setBadge('camera', 'fail', 'Not reachable');
      setDot('camera-dot', 'var(--red)');
      document.getElementById('camera-addr').textContent = data.camera.addr;
      connStatus.camera = false;
    }

    const allOk = connStatus.robot && connStatus.camera;
    document.getElementById('next-1').disabled = !allOk;

    if (!allOk) {
      document.getElementById('conn-note').style.display = 'block';
      document.getElementById('conn-note').textContent =
        'Some connections failed. Enable Dry Run to test without hardware, or fix the connection and re-check.';
    }
  } catch (e) {
    if (dryRun) return;
    document.getElementById('conn-note').style.display = 'block';
    document.getElementById('conn-note').textContent = 'Connection check failed: ' + e;
  }
}

// ── Step 2: Configure ────────────────────────────────────────────────────────
function buildFactorList() {
  const container = document.getElementById('factor-list');
  container.innerHTML = '';
  FACTORS_META.forEach(f => {
    const row = document.createElement('div');
    row.className = 'factor-row';
    row.id = `factor-row-${f.name}`;

    let fixedWidget = '';
    if (f.kind === 'categorical') {
      // Show True/False for boolean factors (levels [0, 1]), numeric otherwise
      const opts = f.is_bool
        ? `<option value="0">False</option><option value="1">True</option>`
        : f.levels.map(v => `<option value="${v}">${v}</option>`).join('');
      fixedWidget = `
        <div class="fixed-input-wrap" id="fixed-wrap-${f.name}" style="display:none">
          <label>Fixed value</label>
          <select id="fixed-val-${f.name}" title="Select a fixed value for ${f.label}">${opts}</select>
        </div>`;
    } else {
      fixedWidget = `
        <div class="fixed-input-wrap" id="fixed-wrap-${f.name}" style="display:none">
          <label>Fixed value</label>
          <input type="number" id="fixed-val-${f.name}" value="${f.default}"
                 step="0.1" min="${f.low}" max="${f.high}"
                 oninput="validateFixedInput('${f.name}',${f.low},${f.high})"
                 title="Enter a fixed value between ${f.low} and ${f.high}">
          <span style="font-size:.75rem;color:var(--red);display:none" id="fixed-err-${f.name}"></span>
        </div>`;
    }

    // Tooltip for known factors
    const tips = {
      aspirate_speed: 'Speed at which the pipette draws liquid into the tip (µL/s). Slower speeds can reduce air bubble formation.',
      dispense_speed: 'Speed at which the pipette pushes liquid out of the tip (µL/s). Slower speeds may improve accuracy for viscous liquids.',
      air_gap:        'When True, a small air gap (2 µL) is aspirated after the liquid to prevent dripping during tip movement.',
      blow_out:       'When True, the pipette performs a blow-out at the destination after dispensing to expel any residual liquid.',
    };
    const tipHtml = tips[f.name]
      ? `<span class="tip" data-tip="${tips[f.name]}">?</span>`
      : '';

    row.innerHTML = `
      <label class="toggle" title="Check to optimize this parameter; uncheck to fix it">
        <input type="checkbox" id="opt-${f.name}" checked onchange="toggleFactor('${f.name}')">
        <span class="tslider"></span>
      </label>
      <div class="factor-meta" style="flex:1">
        <div class="factor-name">${f.label} ${tipHtml}</div>
        <div class="factor-range">${f.range_label}</div>
      </div>
      ${fixedWidget}
    `;
    container.appendChild(row);
  });
}

function toggleFactor(name) {
  const checked = document.getElementById(`opt-${name}`).checked;
  const wrap = document.getElementById(`fixed-wrap-${name}`);
  wrap.style.display = checked ? 'none' : 'flex';
  if (initMode === 'custom') buildCustomTable();
}

function selectInit(mode) {
  initMode = mode;
  document.getElementById('init-algo').classList.toggle('selected', mode === 'algorithm');
  document.getElementById('init-custom').classList.toggle('selected', mode === 'custom');
  document.getElementById('custom-table-wrap').style.display = mode === 'custom' ? 'block' : 'none';
  if (mode === 'custom') buildCustomTable();
}

function getActiveFactors() {
  return FACTORS_META.filter(f => document.getElementById(`opt-${f.name}`).checked);
}

function buildCustomTable() {
  const active = getActiveFactors();
  document.getElementById('custom-header').innerHTML =
    active.map(f => `<th>${f.label}</th>`).join('') + '<th></th>';
  const existingRows = document.getElementById('custom-body').querySelectorAll('tr').length;
  if (existingRows === 0) addCustomRow();
}

let customRowId = 0;
function addCustomRow() {
  const active = getActiveFactors();
  const body = document.getElementById('custom-body');
  const rid = customRowId++;
  const tr = document.createElement('tr');
  tr.id = `crow-${rid}`;
  tr.innerHTML = active.map(f => {
    if (f.kind === 'categorical') {
      const opts = f.is_bool
        ? `<option value="0">False</option><option value="1">True</option>`
        : f.levels.map(v => `<option value="${v}">${v}</option>`).join('');
      return `<td><select class="custom-input" data-factor="${f.name}">${opts}</select></td>`;
    }
    return `<td><input type="number" class="custom-input" data-factor="${f.name}"
              value="${f.default}" step="0.1" min="${f.low}" max="${f.high}"
              title="${f.low} – ${f.high}"></td>`;
  }).join('') + `<td><button onclick="document.getElementById('crow-${rid}').remove()"
    style="background:none;border:none;color:var(--red);cursor:pointer;font-size:0.9rem"
    title="Remove this experiment row">Remove</button></td>`;
  body.appendChild(tr);
}

function getCustomRows() {
  const rows = [];
  document.querySelectorAll('#custom-body tr').forEach(tr => {
    const row = {};
    tr.querySelectorAll('.custom-input').forEach(inp => {
      row[inp.dataset.factor] = parseFloat(inp.value);
    });
    rows.push(row);
  });
  return rows;
}

document.getElementById('lights-toggle').addEventListener('change', function() { lights = this.checked; });

// ── Step 4: Calibration ───────────────────────────────────────────────────────
async function loadCalibration() {
  document.getElementById('cal-list').innerHTML = `
    <div class="cal-item">
      <div class="cal-indicator" style="background:var(--yellow)"></div>
      <div class="cal-info"><div class="cal-name"><span class="spinner"></span>Querying OT-2 calibration API…</div></div>
    </div>`;
  document.getElementById('cal-note').style.display = 'none';
  document.getElementById('cal-warn').style.display = 'none';

  if (dryRun) {
    document.getElementById('cal-list').innerHTML = `
      <div class="cal-item">
        <div class="cal-indicator" style="background:var(--accent)"></div>
        <div class="cal-info">
          <div class="cal-name">Calibration check skipped (Dry Run)</div>
          <div class="cal-detail">No robot connection in dry run mode</div>
        </div>
        <span class="cal-status" style="color:var(--accent)">SKIPPED</span>
      </div>`;
    document.getElementById('next-4').disabled = false;
    return;
  }

  try {
    const robotIpOverride = document.getElementById('robot-ip-input').value.trim();
    const res = await fetch('/api/check-calibration', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ robot_ip: robotIpOverride }),
    });
    calData = await res.json();

    const list = document.getElementById('cal-list');
    list.innerHTML = '';

    calData.items.forEach(item => {
      const color = item.status === 'ok' ? 'var(--green)' : item.status === 'warn' ? 'var(--yellow)' : 'var(--red)';
      const statusText = item.status === 'ok' ? 'PASS' : item.status === 'warn' ? 'WARN' : 'FAIL';
      const div = document.createElement('div');
      div.className = 'cal-item';
      div.innerHTML = `
        <div class="cal-indicator" style="background:${color}"></div>
        <div class="cal-info">
          <div class="cal-name">${item.name}</div>
          <div class="cal-detail">${item.detail}</div>
        </div>
        <span class="cal-status" style="color:${color}">${statusText}</span>
      `;
      list.appendChild(div);
    });

    const hasError = calData.items.some(i => i.status === 'error');
    const hasWarn  = calData.items.some(i => i.status === 'warn');
    if (hasError) {
      document.getElementById('cal-warn').style.display = 'block';
      document.getElementById('cal-warn').textContent =
        'Calibration errors detected. Results may be inaccurate. Consider re-running calibration in the Opentrons app before proceeding.';
    } else if (hasWarn) {
      document.getElementById('cal-note').style.display = 'block';
      document.getElementById('cal-note').textContent =
        'Some calibration data is missing or outdated. You can still proceed, but consider recalibrating for best results.';
    } else {
      document.getElementById('cal-note').style.display = 'block';
      document.getElementById('cal-note').textContent = 'All calibration checks passed.';
    }
    document.getElementById('next-4').disabled = false;
  } catch (e) {
    document.getElementById('cal-list').innerHTML = `
      <div class="cal-item">
        <div class="cal-indicator" style="background:var(--red)"></div>
        <div class="cal-info">
          <div class="cal-name">Could not retrieve calibration data</div>
          <div class="cal-detail">${e}</div>
        </div>
      </div>`;
    document.getElementById('cal-warn').style.display = 'block';
    document.getElementById('cal-warn').textContent = 'Could not connect to robot API. You may still proceed.';
    document.getElementById('next-4').disabled = false;
  }
}

// ── Step 5: Review ────────────────────────────────────────────────────────────
function buildReview() {
  const active = getActiveFactors();
  const fixed = FACTORS_META.filter(f => !document.getElementById(`opt-${f.name}`).checked)
    .map(f => {
      const el = document.getElementById(`fixed-val-${f.name}`);
      let val = el ? el.value : '?';
      if (f.is_bool) val = (parseFloat(val) === 1) ? 'True' : 'False';
      return `${f.label} = ${val}`;
    });

  const nInit = document.getElementById('n-initial').value;
  const maxIter = document.getElementById('max-iter').value;
  const convWin = document.getElementById('conv-window').value;
  const convTol = document.getElementById('conv-tol').value;

  const grid = document.getElementById('review-grid');
  grid.innerHTML = `
    <div class="review-item">
      <div class="r-label">Run Mode</div>
      <div class="r-value">${dryRun ? 'Dry Run' : 'Live'}</div>
      <div class="r-sub">${dryRun ? 'Simulated accuracy, no robot' : 'Real OT-2 hardware run'}</div>
    </div>
    <div class="review-item">
      <div class="r-label">OT-2 Deck Lights</div>
      <div class="r-value">${lights ? 'On' : 'Off'}</div>
    </div>
    <div class="review-item">
      <div class="r-label">Optimized Parameters</div>
      <div class="r-value" style="font-size:.88rem">${active.map(f => f.label).join(', ') || 'None'}</div>
    </div>
    <div class="review-item">
      <div class="r-label">Fixed Parameters</div>
      <div class="r-value" style="font-size:.88rem">${fixed.length ? fixed.join(', ') : 'None — all optimized'}</div>
    </div>
    <div class="review-item">
      <div class="r-label">Initial Design</div>
      <div class="r-value">${initMode === 'algorithm' ? 'LHS Algorithm' : 'Custom'}</div>
      <div class="r-sub">${nInit} initial points</div>
    </div>
    <div class="review-item">
      <div class="r-label">Stopping Criteria</div>
      <div class="r-value">Max ${maxIter} experiments</div>
      <div class="r-sub">Converge if &lt;${convTol}% gain over ${convWin} BO rounds</div>
    </div>
    <div class="review-item" style="grid-column:1/-1">
      <div class="r-label">Protocol files saved to</div>
      <div class="r-value">
        <span class="path-link" onclick="openPath('__PROTOCOLS_DIR__')"
              title="Click to open this folder in Finder / Explorer">
          __PROTOCOLS_DIR__
        </span>
      </div>
    </div>
  `;
}

async function openPath(path) {
  try {
    await fetch('/api/open-path', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
  } catch (e) { /* ignore */ }
}

async function launchCampaign() {
  document.getElementById('start-btn').disabled = true;
  document.getElementById('start-btn').innerHTML = '<span class="spinner"></span>Launching…';

  const active = getActiveFactors();
  const fixedParams = {};
  FACTORS_META.forEach(f => {
    if (!document.getElementById(`opt-${f.name}`).checked) {
      const el = document.getElementById(`fixed-val-${f.name}`);
      fixedParams[f.name] = parseFloat(el ? el.value : 0);
    }
  });

  const config = {
    dry_run: dryRun,
    lights: lights,
    active_factors: active.map(f => f.name),
    fixed_params: fixedParams,
    initial_mode: initMode,
    n_initial: parseInt(document.getElementById('n-initial').value),
    custom_initial: initMode === 'custom' ? getCustomRows() : [],
    max_iterations: parseInt(document.getElementById('max-iter').value),
    convergence_window: parseInt(document.getElementById('conv-window').value),
    convergence_tol: parseFloat(document.getElementById('conv-tol').value),
    robot_ip: document.getElementById('robot-ip-input').value.trim() || null,
    camera_ip: document.getElementById('camera-ip-input').value.trim(),
    camera_port: parseInt(document.getElementById('camera-port-input').value) || 8080,
    camera_index: parseInt(document.getElementById('camera-index-input').value) || 0,
  };

  try {
    const res = await fetch('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (res.ok) {
      const note = document.getElementById('launch-note');
      note.textContent = 'Campaign started. Redirecting to dashboard in 3 seconds…';
      note.style.cssText = 'background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);border-radius:8px;padding:12px 16px;color:#86efac;margin-top:14px;font-size:.9rem;font-weight:600';
      setTimeout(() => { window.location.href = '/'; }, 3000);
    }
  } catch (e) {
    document.getElementById('start-btn').disabled = false;
    document.getElementById('start-btn').textContent = 'Start Campaign';
    alert('Failed to start: ' + e);
  }
}

// ── Init ────────────────────────────────────────────────────────────────────
buildFactorList();
checkConnections();
</script>
</body>
</html>
"""


# ── Wizard server ─────────────────────────────────────────────────────────────

class SetupWizard:
    """
    Multi-step setup wizard served as a local web page.

    Call flow:
        wizard = SetupWizard(port, robot_ips, camera_ip, camera_port)
        wizard.start()           # launches HTTP server + opens browser
        config = wizard.wait_for_start()   # blocks until user clicks "Start Campaign"
        wizard.stop()
        # → pass config to run_campaign()
    """

    def __init__(
        self,
        port: int,
        robot_ips: list,
        camera_ip: str,
        camera_port: int,
        factors_meta: list,
        protocols_dir: str = "",
        dry_run_forced: bool = False,
        camera_index: int = 0,
    ):
        self.port = port
        self.robot_ips = robot_ips
        self.camera_ip = camera_ip
        self.camera_port = camera_port
        self.camera_index = camera_index
        self.factors_meta = factors_meta
        self.protocols_dir = protocols_dir
        self.dry_run_forced = dry_run_forced
        self._resolved_robot_ip: str | None = None
        self._cam_proc = None   # subprocess handle for auto-started camera server

        self._config: dict | None = None
        self._ready = threading.Event()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ── HTTP server ───────────────────────────────────────────────────────────

    def start(self):
        wizard = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass   # silence request logs

            def _send_json(self, data: dict, status: int = 200):
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_html(self, html: str):
                body = html.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/" or self.path == "/wizard":
                    self._send_html(wizard._build_html())
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length) if length else b""

                if self.path == "/api/check-connections":
                    try:
                        req = json.loads(body.decode()) if body else {}
                    except Exception:
                        req = {}
                    self._send_json(wizard._check_connections(req))

                elif self.path == "/api/check-calibration":
                    try:
                        req = json.loads(body.decode()) if body else {}
                    except Exception:
                        req = {}
                    self._send_json(wizard._check_calibration(req))

                elif self.path == "/api/open-path":
                    try:
                        data = json.loads(body.decode())
                        path = data.get("path", "")
                        wizard._open_path(path)
                        self._send_json({"ok": True})
                    except Exception as e:
                        self._send_json({"ok": False, "error": str(e)})

                elif self.path == "/api/start":
                    try:
                        config = json.loads(body.decode())
                        wizard._config = config
                        wizard._ready.set()
                        self._send_json({"ok": True})
                    except Exception as e:
                        self._send_json({"ok": False, "error": str(e)}, 400)

                else:
                    self.send_response(404)
                    self.end_headers()

        class _ReuseServer(HTTPServer):
            allow_reuse_address = True
            def server_bind(self):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                super().server_bind()

        port = self.port
        for _ in range(10):
            try:
                self._server = _ReuseServer(("", port), _Handler)
                self.port = port
                break
            except OSError:
                port += 1
        else:
            raise RuntimeError(f"Could not bind to any port starting at {self.port}")

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        url = f"http://localhost:{self.port}/"
        print(f"\n  Setup Wizard -> {url}")
        try:
            webbrowser.open(url)
        except Exception:
            pass

    def wait_for_start(self) -> dict:
        """Block until the user clicks 'Start Campaign' in the wizard."""
        print("  Waiting for wizard configuration…", flush=True)
        self._ready.wait()
        return self._config

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server = None

    # ── HTML builder ──────────────────────────────────────────────────────────

    def _build_html(self) -> str:
        meta_json = json.dumps(self.factors_meta)
        html = _WIZARD_HTML.replace("__FACTORS_META__", meta_json)
        html = html.replace("__PROTOCOLS_DIR__", self.protocols_dir.replace("\\", "/"))
        html = html.replace("__CAMERA_IP__", self.camera_ip)
        html = html.replace("__CAMERA_PORT__", str(self.camera_port))
        html = html.replace("__CAMERA_INDEX__", str(self.camera_index))
        if self.dry_run_forced:
            html = html.replace(
                'id="dry-run-toggle">',
                'id="dry-run-toggle" checked disabled>'
            )
        return html

    # ── Open folder ──────────────────────────────────────────────────────────

    def _open_path(self, path: str):
        """Open a file or folder in the system file manager."""
        try:
            p = Path(path)
            target = p if p.is_dir() else p.parent
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            elif sys.platform == "win32":
                subprocess.Popen(["explorer", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception:
            pass

    # ── Connection check ─────────────────────────────────────────────────────

    def _check_connections(self, req: dict = None) -> dict:
        import time as _time
        req = req or {}
        robot_ip_override = (req.get("robot_ip") or "").strip()
        camera_ip    = (req.get("camera_ip") or self.camera_ip).strip() or self.camera_ip
        camera_port  = int(req.get("camera_port") or self.camera_port)
        camera_index = int(req.get("camera_index") if req.get("camera_index") is not None
                          else self.camera_index)

        # Build ordered list of robot IPs: user override first, then defaults
        if robot_ip_override:
            robot_ips = [robot_ip_override] + [ip for ip in self.robot_ips if ip != robot_ip_override]
        else:
            robot_ips = self.robot_ips

        result = {
            "robot":  {"ok": False, "ip": None},
            "camera": {"ok": False, "addr": f"{camera_ip}:{camera_port}"},
        }

        # ── Robot check ───────────────────────────────────────────────────────
        for ip in robot_ips:
            try:
                url = f"http://{ip}:31950/health"
                with urllib.request.urlopen(url, timeout=3) as r:
                    if r.status == 200:
                        result["robot"] = {"ok": True, "ip": ip}
                        self._resolved_robot_ip = ip
                        break
            except Exception:
                continue

        # ── Camera check (always verify via localhost — server runs on this machine) ──
        def _cam_alive(ip: str, port: int, timeout: float = 2.0) -> bool:
            try:
                with urllib.request.urlopen(f"http://{ip}:{port}/health", timeout=timeout) as r:
                    return r.status == 200
            except Exception:
                return False

        # First check if it's already running locally
        cam_ok = _cam_alive("127.0.0.1", camera_port)

        # If not running, auto-start camera/server.py
        if not cam_ok and self._cam_proc is None:
            server_script = Path(__file__).parent.parent / "camera" / "server.py"
            if server_script.exists():
                cmd = [sys.executable, str(server_script),
                       "--camera-index", str(camera_index),
                       "--port",         str(camera_port)]
                try:
                    self._cam_proc = subprocess.Popen(cmd)
                    # Wait up to 8 s for server to come up
                    for _ in range(16):
                        _time.sleep(0.5)
                        if _cam_alive("127.0.0.1", camera_port):
                            cam_ok = True
                            break
                except Exception:
                    pass

        result["camera"]["ok"] = cam_ok

        return result

    # ── Calibration check ─────────────────────────────────────────────────────

    def _check_calibration(self, req: dict = None) -> dict:
        req = req or {}
        robot_ip_override = (req.get("robot_ip") or "").strip()

        # Use resolved IP from connection check if available, then override, then defaults
        if self._resolved_robot_ip:
            robot_ips = [self._resolved_robot_ip]
        elif robot_ip_override:
            robot_ips = [robot_ip_override] + [ip for ip in self.robot_ips if ip != robot_ip_override]
        else:
            robot_ips = self.robot_ips

        items = []
        robot_ip = None

        for ip in robot_ips:
            try:
                url = f"http://{ip}:31950/health"
                with urllib.request.urlopen(url, timeout=3) as r:
                    if r.status == 200:
                        robot_ip = ip
                        break
            except Exception:
                continue

        if not robot_ip:
            return {"items": [{"name": "Robot unreachable", "detail": "No OT-2 found on network", "status": "error"}]}

        base = f"http://{robot_ip}:31950"

        try:
            with urllib.request.urlopen(f"{base}/calibration/status", timeout=5) as r:
                data = json.loads(r.read().decode())
            deckCal = data.get("deckCalibration", {})
            ok = deckCal.get("status", "") == "OK"
            items.append({
                "name": "Deck Calibration",
                "detail": f"Status: {deckCal.get('status', 'unknown')}",
                "status": "ok" if ok else "warn",
            })
        except Exception as e:
            items.append({"name": "Deck Calibration", "detail": str(e), "status": "warn"})

        try:
            with urllib.request.urlopen(f"{base}/calibration/pipette_offset", timeout=5) as r:
                offsets = json.loads(r.read().decode())
            has_left = any(o.get("mount") == "left" for o in offsets.get("data", []))
            items.append({
                "name": "Pipette Offset Calibration (left mount)",
                "detail": "Calibration data found" if has_left else "No calibration data for left mount",
                "status": "ok" if has_left else "warn",
            })
        except Exception as e:
            items.append({"name": "Pipette Offset Calibration", "detail": str(e), "status": "warn"})

        try:
            with urllib.request.urlopen(f"{base}/calibration/tip_length", timeout=5) as r:
                tips = json.loads(r.read().decode())
            count = len(tips.get("data", []))
            items.append({
                "name": "Tip Length Calibration",
                "detail": f"{count} tip calibration record(s) found" if count else "No tip length calibration found",
                "status": "ok" if count else "warn",
            })
        except Exception as e:
            items.append({"name": "Tip Length Calibration", "detail": str(e), "status": "warn"})

        return {"items": items}


# ── Factor metadata builder ───────────────────────────────────────────────────

def build_factors_meta(factors) -> list:
    """Convert campaign Factor objects → JSON-serialisable dicts for the wizard."""
    meta = []
    for f in factors:
        is_bool = f.kind == "categorical" and sorted(f.levels) == [0.0, 1.0]

        if f.kind == "categorical":
            if is_bool:
                range_label = "Options: False, True"
            else:
                range_label = "Options: " + ", ".join(str(v) for v in f.levels)
            default = f.levels[0]
            levels = [float(v) for v in f.levels]
        else:
            range_label = f"{f.low} \u2013 {f.high} \u00b5L/s"
            default = round((f.low + f.high) / 2, 2)
            levels = []

        meta.append({
            "name":        f.name,
            "label":       f.name.replace("_", " ").title(),
            "kind":        f.kind,
            "is_bool":     is_bool,
            "low":         float(f.low) if f.kind != "categorical" else 0,
            "high":        float(f.high) if f.kind != "categorical" else 1,
            "default":     float(default),
            "levels":      levels,
            "range_label": range_label,
        })
    return meta
