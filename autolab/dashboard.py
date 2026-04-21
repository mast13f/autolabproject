"""
Campaign Dashboard
==================
Live HTML dashboard served at http://localhost:PORT/ with:
  - Real-time accuracy chart and experiment log
  - Pause / Resume / Stop buttons
  - Per-iteration user confirmation gate (awaiting_confirmation state)
  - Per-iteration protocol file tracking
  - QC checks display

The dashboard auto-refreshes every 3 seconds. Buttons POST to /control
on the same server (no file:// limitations).

Usage:
    from campaign_dashboard import CampaignDashboard
    db = CampaignDashboard(port=9999)
    db.start()
    db.update(optimizer, ...)   # call after each iteration
    state = db.get_state()      # "running" | "paused" | "stopped" | "completed"
                                # | "awaiting_confirmation"
"""

import json
import statistics
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

DEFAULT_PORT = 9999


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _status_class(state: str) -> str:
    return {
        "running":               "status-running",
        "paused":                "status-paused",
        "stop_now":              "status-stopping",
        "stopped":               "status-stopped",
        "completed":             "status-converged",
        "awaiting_confirmation": "status-awaiting",
    }.get(state, "status-running")


def _mean_acc(experiments: list) -> float:
    if not experiments:
        return 0.0
    return sum(e["accuracy"] for e in experiments) / len(experiments)


def _std_acc(experiments: list) -> float:
    if len(experiments) < 2:
        return 0.0
    return statistics.stdev(e["accuracy"] for e in experiments)


def _latest_ei(experiments: list) -> str:
    for e in reversed(experiments):
        if e.get("ei") is not None:
            return f"{e['ei']:.4f}"
    return "N/A"


def _next_params_html(suggestion: Optional[dict]) -> str:
    if not suggestion:
        return '<div style="color:#64748b;padding:20px;text-align:center">Waiting for first BO iteration…</div>'
    html = '<div class="param-grid">'
    for k, v in suggestion.items():
        html += (
            f'<div class="param-cell">'
            f'<div class="param-name">{k.replace("_"," ")}</div>'
            f'<div class="param-val">{v}</div>'
            f'</div>'
        )
    html += '</div>'
    return html


def _current_params_html(params: Optional[dict]) -> str:
    if not params:
        return '<div style="color:#64748b;padding:20px;text-align:center">No experiment running yet…</div>'
    html = '<div class="param-grid">'
    for k, v in params.items():
        html += (
            f'<div class="param-cell">'
            f'<div class="param-name">{k.replace("_"," ")}</div>'
            f'<div class="param-val" style="color:#fbbf24">{v}</div>'
            f'</div>'
        )
    html += '</div>'
    return html


def _best_params_html(best: dict) -> str:
    html = '<div class="param-grid" style="margin-top:8px">'
    for k, v in best.get("params", {}).items():
        html += (
            f'<div class="param-cell">'
            f'<div class="param-name">{k.replace("_"," ")}</div>'
            f'<div class="param-val" style="color:#34d399">{v}</div>'
            f'</div>'
        )
    html += '</div>'
    return html


def _qc_html(qc_pre: list) -> str:
    if not qc_pre:
        return ""
    items = ""
    for qc in qc_pre:
        icon_cls = "qc-pass" if qc.get("passed") else "qc-fail"
        icon = "&#10003;" if qc.get("passed") else "&#10007;"
        items += f"""
          <div class="qc-item">
            <div class="qc-icon {icon_cls}">{icon}</div>
            <div>
              <div class="qc-name">{qc["name"]}</div>
              <div class="qc-value">{qc["value"]}</div>
              <div class="qc-rationale">{qc.get("rationale","")}</div>
            </div>
          </div>"""
    return f'<div class="section"><h2>Pre-Experiment QC</h2><div class="qc-grid">{items}</div></div>'


def _table_rows(experiments: list, best_iter: int, factor_names: list) -> str:
    rows = ""
    for e in reversed(experiments):   # newest first
        is_best = e["iteration"] == best_iter
        rc = ' class="best-row"' if is_best else ""
        ph_cls = "phase-initial" if e["phase"] == "INITIAL" else "phase-bo"
        ei_str = f"{e['ei']:.4f}" if e.get("ei") is not None else "—"
        acc = e["accuracy"]
        bar_col = "#34d399" if is_best else ("#7dd3fc" if e["phase"] == "INITIAL" else "#c4b5fd")

        factor_cells = "".join(
            f"<td>{e['params'].get(n, '—')}</td>" for n in factor_names
        )
        proto = e.get("protocol_file", "")
        proto_label = f'<span title="{proto}" style="font-size:11px;color:#64748b">{proto[-28:] if len(proto) > 28 else proto or "—"}</span>'

        rows += f"""<tr{rc}>
          <td>{e['iteration']}</td>
          <td><span class="phase-badge {ph_cls}">{e['phase']}</span></td>
          {factor_cells}
          <td class="bar-cell">
            <div class="bar-fill" style="width:{acc:.1f}%;background:{bar_col}"></div>
            {acc:.1f}%
          </td>
          <td>{ei_str}</td>
          <td>{e.get('best_so_far', 0):.1f}%</td>
          <td>{proto_label}</td>
        </tr>\n"""
    return rows


def generate_html(data: dict) -> str:
    """Render the full dashboard HTML from the data dict."""
    if not data:
        return """<!DOCTYPE html><html><body style="background:#0f172a;color:#e2e8f0;
          font-family:sans-serif;display:flex;align-items:center;justify-content:center;
          height:100vh;font-size:24px">
          <div>Starting campaign… <meta http-equiv="refresh" content="2"></div></body></html>"""

    experiments   = data.get("experiments", [])
    config        = data.get("config", {})
    status        = data.get("status", {})
    best          = data.get("best", {"accuracy": 0, "iteration": 0, "params": {}})
    next_sug      = data.get("next_suggestion")
    current_params = data.get("current_params") or {}
    qc_pre       = data.get("qc_pre_experiment", [])
    factor_names = config.get("factors", [])
    state        = status.get("state", "running")
    stop_reason  = status.get("stop_reason", "")
    is_dry_run   = data.get("dry_run", False)

    # Button visibility
    show_pause    = "" if state == "running"               else "display:none"
    show_resume   = "" if state == "paused"                else "display:none"
    show_stop     = "" if state in ("running", "paused", "awaiting_confirmation") else "display:none"
    show_stop_now = "" if state in ("running", "paused", "stop_now") else "display:none"
    show_confirm  = "" if state == "awaiting_confirmation" else "display:none"

    # Robot status data
    rs = data.get("robot_status", {})
    rs_step    = rs.get("step", "Idle")
    rs_status  = rs.get("ot2_status", "idle").lower()
    rs_run_id  = rs.get("run_id", "")
    rs_iter    = rs.get("iteration")
    dot_cls    = "dot-running" if rs_status == "running" else ("dot-error" if rs_status in ("failed","stopped") else "dot-idle")

    # Table header for factors
    factor_th = "".join(
        f'<th>{n.replace("_"," ").title()}</th>' for n in factor_names
    )

    # Chart data
    accs = [e["accuracy"] for e in experiments]
    bsf, b = [], -1
    for a in accs:
        b = max(b, a)
        bsf.append(b)
    chart_colors = [
        "#7dd3fc" if e["phase"] == "INITIAL" else "#c4b5fd"
        for e in experiments
    ]
    chart_json = json.dumps({
        "accuracy":    accs,
        "best_so_far": bsf,
        "colors":      chart_colors,
        "n_initial":   config.get("n_initial", 5),
    })

    stop_banner = (
        f'<div class="stop-banner">{stop_reason}</div>'
        if stop_reason else ""
    )

    # Pre-build camera block — backslashes not allowed inside f-string expressions in Python < 3.12
    if is_dry_run:
        _camera_feed_html = (
            '<div class="camera-placeholder">'
            '<div class="icon">&#128247;</div>'
            '<div class="msg">Dry Run — No Camera Feed</div>'
            '<div class="sub">Images are not captured in dry run mode</div>'
            '</div>'
        )
    else:
        _onerror = "this.style.display='none';document.getElementById('cam-ph').style.display='flex'"
        _camera_feed_html = (
            '<div class="camera-frame">'
            '<img id="cam-img" src="/camera-feed" alt="Latest captured image"'
            f' onerror="{_onerror}">'
            '<div id="cam-ph" class="camera-placeholder" style="display:none">'
            '<div class="icon">&#128247;</div>'
            '<div class="msg">Waiting for first image</div>'
            '<div class="sub">Images appear here after the first experiment completes</div>'
            '</div>'
            '</div>'
            '<script>'
            '(function(){'
            'var img=document.getElementById("cam-img");'
            'var ph=document.getElementById("cam-ph");'
            'if(!img)return;'
            'function refresh(){'
            'var next=new Image();'
            'next.onload=function(){img.src=next.src;img.style.display="block";ph.style.display="none";};'
            'next.onerror=function(){};'
            'next.src="/camera-feed?t="+Date.now();'
            '}'
            'setInterval(refresh,2000);'
            '})();'
            '</script>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="3">
<title>AutoLab Campaign</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
        background:#0f172a;color:#e2e8f0;padding:20px;min-height:100vh}}
  h2{{font-size:13px;color:#94a3b8;text-transform:uppercase;letter-spacing:.5px;margin-bottom:14px}}

  /* Header */
  .header{{display:flex;justify-content:space-between;align-items:center;
           margin-bottom:22px;padding-bottom:14px;border-bottom:1px solid #334155;flex-wrap:wrap;gap:10px}}
  .header h1{{font-size:22px;color:#f8fafc;font-weight:700}}
  .header-right{{display:flex;align-items:center;gap:10px;flex-wrap:wrap}}

  /* Status badge */
  .status{{padding:5px 14px;border-radius:20px;font-size:12px;font-weight:600}}
  .status-running{{background:#065f46;color:#6ee7b7}}
  .status-paused{{background:#78350f;color:#fcd34d}}
  .status-stopped{{background:#7c2d12;color:#fca5a5}}
  .status-converged{{background:#1e3a5f;color:#7dd3fc}}
  .status-awaiting{{background:#4c1d95;color:#c4b5fd}}

  /* Control buttons */
  .btn{{padding:7px 16px;border-radius:8px;border:none;cursor:pointer;
        font-size:13px;font-weight:600;transition:opacity .15s}}
  .btn:hover{{opacity:.82}}
  .btn-pause{{background:#d97706;color:#fff}}
  .btn-resume{{background:#059669;color:#fff}}
  .btn-stop{{background:#dc2626;color:#fff}}
  .btn-confirm{{background:#7c3aed;color:#fff;font-size:15px;padding:12px 28px;border-radius:10px}}
  .confirm-panel{{background:#1e1338;border:2px solid #7c3aed;border-radius:12px;padding:24px;
                   margin-bottom:22px;text-align:center}}
  .confirm-panel h2{{color:#c4b5fd;font-size:16px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}}
  .confirm-panel .confirm-msg{{color:#a78bfa;font-size:13px;margin-bottom:16px;line-height:1.6}}

  .stop-banner{{background:#7c2d12;color:#fca5a5;padding:11px 18px;
                border-radius:8px;margin-bottom:16px;font-size:13px}}

  /* Stat cards */
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-bottom:22px}}
  .card{{background:#1e293b;border-radius:12px;padding:18px;border:1px solid #334155}}
  .card .lbl{{font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}}
  .card .val{{font-size:26px;font-weight:700;color:#f8fafc}}
  .card .sub{{font-size:11px;color:#64748b;margin-top:3px}}
  .card.hi{{border-color:#3b82f6}}.card.hi .val{{color:#60a5fa}}
  .card.gr{{border-color:#10b981}}.card.gr .val{{color:#34d399}}

  /* Two-column panels */
  .panels{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:22px}}
  @media(max-width:860px){{.panels{{grid-template-columns:1fr}}}}
  .panel{{background:#1e293b;border-radius:12px;padding:18px;border:1px solid #334155}}

  /* Parameter grid */
  .param-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
  .param-cell{{background:#0f172a;border-radius:8px;padding:10px;border:1px solid #334155}}
  .param-name{{font-size:10px;color:#64748b;margin-bottom:2px;text-transform:uppercase;letter-spacing:.3px}}
  .param-val{{font-size:19px;font-weight:600;color:#fbbf24}}

  /* Stopping criteria */
  .criteria{{list-style:none}}
  .criteria li{{padding:9px 11px;margin-bottom:6px;border-radius:8px;
                background:#0f172a;border:1px solid #334155;font-size:13px}}
  .tag{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;
        font-weight:600;margin-right:6px}}
  .tag-active{{background:#065f46;color:#6ee7b7}}
  .tag-done{{background:#7c2d12;color:#fdba74}}

  /* Chart */
  .section{{background:#1e293b;border-radius:12px;padding:18px;
            border:1px solid #334155;margin-bottom:22px}}
  canvas{{width:100%!important;height:280px!important}}

  /* Table */
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  thead th{{text-align:left;padding:9px 11px;color:#94a3b8;font-weight:600;
            border-bottom:2px solid #334155;font-size:11px;text-transform:uppercase;
            letter-spacing:.4px;position:sticky;top:0;background:#1e293b}}
  tbody td{{padding:9px 11px;border-bottom:1px solid #0f172a}}
  tbody tr{{background:#0f172a}} tbody tr:hover{{background:#1e293b}}
  tbody tr.best-row{{background:#052e16}} tbody tr.best-row td{{color:#34d399;font-weight:600}}
  .phase-badge{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:600}}
  .phase-initial{{background:#1e3a5f;color:#7dd3fc}}
  .phase-bo{{background:#3b1f6e;color:#c4b5fd}}
  .bar-cell{{position:relative}}
  .bar-fill{{position:absolute;left:0;top:0;bottom:0;opacity:.15;border-radius:4px}}

  /* QC */
  .qc-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}}
  .qc-item{{background:#0f172a;border-radius:8px;padding:12px;border:1px solid #334155;
            display:flex;align-items:flex-start;gap:10px}}
  .qc-icon{{width:30px;height:30px;border-radius:7px;display:flex;align-items:center;
             justify-content:center;font-size:15px;flex-shrink:0}}
  .qc-pass{{background:#065f46}} .qc-fail{{background:#7c2d12}}
  .qc-name{{font-size:12px;font-weight:600;color:#e2e8f0;margin-bottom:2px}}
  .qc-value{{font-size:11px;color:#94a3b8}}
  .qc-rationale{{font-size:10px;color:#64748b;margin-top:3px;line-height:1.4}}

  .footer{{text-align:center;padding:14px;color:#475569;font-size:11px;margin-top:14px}}

  /* Tooltips */
  .tip{{display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;
        border-radius:50%;background:#334155;color:#94a3b8;font-size:9px;font-weight:700;
        cursor:help;position:relative;vertical-align:middle;margin-left:4px;flex-shrink:0}}
  .tip::after{{content:attr(data-tip);position:absolute;bottom:calc(100% + 8px);left:50%;
               transform:translateX(-50%);white-space:normal;min-width:200px;max-width:300px;
               background:#1e293b;border:1px solid #475569;border-radius:8px;padding:9px 12px;
               font-size:11px;color:#e2e8f0;z-index:999;pointer-events:none;line-height:1.5;
               opacity:0;transition:opacity .15s;font-weight:400;text-transform:none;
               letter-spacing:0;box-shadow:0 4px 20px rgba(0,0,0,.5)}}
  .tip::before{{content:'';position:absolute;bottom:calc(100% + 3px);left:50%;
                transform:translateX(-50%);border:5px solid transparent;
                border-top-color:#475569;z-index:1000;pointer-events:none;
                opacity:0;transition:opacity .15s}}
  .tip:hover::after,.tip:hover::before{{opacity:1}}

  /* Camera feed */
  .camera-frame{{background:#0f172a;border-radius:10px;overflow:hidden;
                 display:flex;align-items:center;justify-content:center;min-height:220px}}
  .camera-frame img{{max-width:100%;max-height:420px;display:block;border-radius:8px}}
  .camera-placeholder{{display:flex;flex-direction:column;align-items:center;
                        justify-content:center;gap:10px;padding:48px;color:#475569}}
  .camera-placeholder .icon{{font-size:36px;opacity:.5}}
  .camera-placeholder .msg{{font-size:14px;font-weight:600}}
  .camera-placeholder .sub{{font-size:11px}}

  /* Robot status */
  .status-stopping{{background:#6b21a8;color:#e9d5ff}}
  .robot-section{{background:#1e293b;border-radius:12px;padding:18px;border:1px solid #334155;margin-bottom:22px}}
  .robot-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:10px}}
  .robot-cell{{background:#0f172a;border-radius:8px;padding:14px;border:1px solid #334155}}
  .robot-label{{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.4px;margin-bottom:6px}}
  .robot-step{{font-size:15px;font-weight:600;color:#e2e8f0;line-height:1.3}}
  .ot2-badge{{display:inline-block;padding:4px 10px;border-radius:6px;font-size:12px;font-weight:600;margin-top:4px}}
  .ot2-running{{background:#065f46;color:#6ee7b7}}
  .ot2-uploading{{background:#1e3a5f;color:#7dd3fc}}
  .ot2-succeeded{{background:#14532d;color:#86efac}}
  .ot2-failed{{background:#7c2d12;color:#fca5a5}}
  .ot2-stopped{{background:#78350f;color:#fcd34d}}
  .ot2-idle{{background:#1e293b;color:#64748b;border:1px solid #334155}}
  .ot2-paused{{background:#78350f;color:#fcd34d}}
  .step-dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}}
  .dot-running{{background:#34d399;animation:pulse 1.5s infinite}}
  .dot-idle{{background:#64748b}}
  .dot-error{{background:#f87171}}
  @keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <h1>AutoLab Campaign Dashboard</h1>
  <div class="header-right">
    <span class="status {_status_class(state)}">{'AWAITING CONFIRMATION' if state == 'awaiting_confirmation' else state.upper()}</span>
    <button class="btn btn-pause"   style="{show_pause}"  onclick="ctrl('pause')">Pause</button>
    <button class="btn btn-resume"  style="{show_resume}" onclick="ctrl('resume')">Resume</button>
    <button class="btn btn-stop"    style="{show_stop}"
            onclick="if(confirm('Finish the current run, then stop?')) ctrl('stop')">
      Stop After Iteration
    </button>
    <button class="btn" style="background:#7c3aed;color:#fff;{show_stop_now}"
            onclick="if(confirm('STOP THE ROBOT NOW?\\nThis immediately halts the current run.')) ctrl('stop_now')">
      &#9632; Stop Now
    </button>
  </div>
</div>

{stop_banner}

<!-- Awaiting confirmation panel -->
<div class="confirm-panel" style="{show_confirm}">
  <h2>&#9888; Waiting for User Confirmation</h2>
  <div class="confirm-msg">
    The previous experiment has finished. You may now replace labware, refill reservoirs,
    or make any deck changes before continuing.<br>
    Review the <strong>Next Suggested Parameters</strong> below, then click the button to proceed.
  </div>
  <button class="btn btn-confirm" onclick="ctrl('confirm_next')">
    &#9654; Confirm &amp; Run Next Experiment
  </button>
</div>

<!-- Stat cards -->
<div class="cards">
  <div class="card hi">
    <div class="lbl">Experiments Run</div>
    <div class="val">{status.get("iteration", 0)}</div>
    <div class="sub">of {config.get("max_iterations", 20)} max</div>
  </div>
  <div class="card gr">
    <div class="lbl">Best Accuracy</div>
    <div class="val">{best["accuracy"]:.1f}%</div>
    <div class="sub">Iteration #{best["iteration"]}</div>
  </div>
  <div class="card">
    <div class="lbl">Current Phase</div>
    <div class="val" style="font-size:18px">{status.get("phase", "—")}</div>
    <div class="sub">{config.get("n_initial", 5)} initial + Bayesian Opt</div>
  </div>
  <div class="card">
    <div class="lbl">Mean Accuracy</div>
    <div class="val">{_mean_acc(experiments):.1f}%</div>
    <div class="sub">across all experiments</div>
  </div>
  <div class="card">
    <div class="lbl">Std Dev <span class="tip" data-tip="Standard deviation of all accuracy scores. High values mean results are inconsistent across experiments.">?</span></div>
    <div class="val">{_std_acc(experiments):.1f}</div>
    <div class="sub">accuracy variability</div>
  </div>
  <div class="card">
    <div class="lbl">Latest EI <span class="tip" data-tip="Expected Improvement — how much better the Bayesian optimizer predicts the next experiment could be over the current best. High EI means high confidence in a big gain.">?</span></div>
    <div class="val">{_latest_ei(experiments)}</div>
    <div class="sub">expected improvement</div>
  </div>
</div>

<!-- Current parameters running -->
<div class="panel" style="margin-bottom:14px;border-color:#d97706">
  <h2 style="display:flex;align-items:center;gap:8px">
    <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{'#f59e0b' if current_params else '#334155'};{'animation:pulse 1.5s infinite' if current_params else ''}"></span>
    Current Parameters Running
    <span class="tip" data-tip="The parameter values being executed by the robot in the current iteration. Updated when each experiment begins.">?</span>
  </h2>
  {_current_params_html(current_params)}
</div>

<!-- Next suggestion + stopping criteria -->
<div class="panels">
  <div class="panel">
    <h2>Next Suggested Parameters <span class="tip" data-tip="The parameter values the Bayesian optimizer recommends for the next experiment, chosen to maximise Expected Improvement over the current best result.">?</span></h2>
    {_next_params_html(next_sug)}
  </div>
  <div class="panel">
    <h2>Stopping Criteria <span class="tip" data-tip="The campaign stops automatically when any one of these criteria is met. Both are evaluated after every completed experiment.">?</span></h2>
    <ul class="criteria">
      <li>
        <span class="tag {'tag-done' if 'converged' in stop_reason.lower() else 'tag-active'}">
          {'TRIGGERED' if 'converged' in stop_reason.lower() else 'WATCHING'}
        </span>
        Convergence <span class="tip" data-tip="Checks the last N Bayesian Optimisation iterations. If the best accuracy improved by less than the tolerance over that window, the optimizer has converged and the campaign stops early.">?</span>
        &nbsp;&lt;&nbsp;{config.get("convergence_tol", 1.0)}% gain over {config.get("convergence_window", 5)} BO rounds
      </li>
      <li>
        <span class="tag {'tag-done' if 'cap' in stop_reason.lower() else 'tag-active'}">
          {'TRIGGERED' if 'cap' in stop_reason.lower() else 'WATCHING'}
        </span>
        Experiment cap: {config.get("max_iterations", 20)} total
      </li>
    </ul>
    <div style="margin-top:14px">
      <h2>Best Parameters Found</h2>
      {_best_params_html(best)}
    </div>
  </div>
</div>

<!-- QC section -->
{_qc_html(qc_pre)}

<!-- Robot Status section -->
<div class="robot-section">
  <h2>Robot Status</h2>
  <div class="robot-grid">
    <div class="robot-cell" style="grid-column:span 2">
      <div class="robot-label">Current Step</div>
      <div class="robot-step">
        <span class="step-dot {dot_cls}"></span>{rs_step}
      </div>
    </div>
    <div class="robot-cell">
      <div class="robot-label">OT-2 Run Status</div>
      <span class="ot2-badge ot2-{rs_status}">{rs_status.upper()}</span>
    </div>
    <div class="robot-cell">
      <div class="robot-label">Current Iteration</div>
      <div class="robot-step">{f"#{rs_iter:03d}" if rs_iter is not None else "—"}</div>
    </div>
    {f'<div class="robot-cell"><div class="robot-label">Run ID</div><div style="font-size:11px;color:#64748b;font-family:monospace">{rs_run_id}</div></div>' if rs_run_id else ""}
  </div>
</div>

<!-- Camera feed -->
<div class="section">
  <h2>Camera Live View{' &nbsp;<span style="font-size:11px;color:#64748b;font-weight:400;text-transform:none">updates every 2 s</span>' if not is_dry_run else ''}</h2>
  {_camera_feed_html}
</div>

<!-- Accuracy chart -->
<div class="section">
  <h2>Accuracy vs. Iteration</h2>
  <canvas id="chart"></canvas>
</div>

<!-- Experiment log table -->
<div class="section">
  <h2>Experiment Log (newest first)</h2>
  <div style="overflow-x:auto">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Phase <span class="tip" data-tip="INITIAL = Latin Hypercube Sampling point chosen before Bayesian optimisation begins. BO = point suggested by the Gaussian Process surrogate model.">?</span></th>
          {factor_th}
          <th>Accuracy</th>
          <th>EI <span class="tip" data-tip="Expected Improvement — the optimizer's predicted gain over the current best before this experiment ran. Only available for BO-phase experiments.">?</span></th>
          <th>Best So Far <span class="tip" data-tip="The highest accuracy achieved in any experiment up to and including this iteration.">?</span></th>
          <th>Protocol File</th>
        </tr>
      </thead>
      <tbody>
        {_table_rows(experiments, best["iteration"], factor_names)}
      </tbody>
    </table>
  </div>
</div>

<div class="footer">
  Auto-refreshes every 3 s &nbsp;&middot;&nbsp;
  Last updated: <span id="ts"></span>
  <script>document.getElementById('ts').textContent=new Date().toLocaleTimeString();</script>
</div>

<script>
/* Control button handler */
function ctrl(action) {{
  fetch('/control', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{action: action}})
  }}).then(() => setTimeout(() => location.reload(), 400));
}}

/* Accuracy chart */
const D = {chart_json};
const cv = document.getElementById('chart');
const ctx = cv.getContext('2d');
const dpr = window.devicePixelRatio || 1;
const rect = cv.parentElement.getBoundingClientRect();
cv.width  = rect.width * dpr;
cv.height = 280 * dpr;
cv.style.width  = rect.width + 'px';
cv.style.height = '280px';
ctx.scale(dpr, dpr);

const W=rect.width, H=280;
const pad={{top:20,right:20,bottom:38,left:50}};
const pw=W-pad.left-pad.right, ph=H-pad.top-pad.bottom;
const n=D.accuracy.length;
const xScale = n > 1 ? pw/(n-1) : pw/2;
const toX = i => pad.left + i*xScale;
const toY = v => pad.top + ph - v/100*ph;

/* Grid */
ctx.strokeStyle='#334155'; ctx.lineWidth=.5;
for (let v=0;v<=100;v+=20) {{
  ctx.beginPath(); ctx.moveTo(pad.left,toY(v)); ctx.lineTo(W-pad.right,toY(v)); ctx.stroke();
  ctx.fillStyle='#64748b'; ctx.font='11px sans-serif'; ctx.textAlign='right';
  ctx.fillText(v+'%', pad.left-6, toY(v)+4);
}}

/* Phase boundary */
if (D.n_initial>0 && D.n_initial<n) {{
  const bx=toX(D.n_initial-.5);
  ctx.strokeStyle='#475569'; ctx.lineWidth=1; ctx.setLineDash([5,5]);
  ctx.beginPath(); ctx.moveTo(bx,pad.top); ctx.lineTo(bx,H-pad.bottom); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#64748b'; ctx.font='10px sans-serif'; ctx.textAlign='center';
  ctx.fillText('Initial | BO', bx, pad.top-4);
}}

/* Best-so-far line */
ctx.strokeStyle='#ef4444'; ctx.lineWidth=2;
ctx.beginPath();
for (let i=0;i<n;i++) i===0?ctx.moveTo(toX(i),toY(D.best_so_far[i])):ctx.lineTo(toX(i),toY(D.best_so_far[i]));
ctx.stroke();

/* Scatter dots */
for (let i=0;i<n;i++) {{
  ctx.beginPath(); ctx.arc(toX(i),toY(D.accuracy[i]),5,0,Math.PI*2);
  ctx.fillStyle=D.colors[i]; ctx.fill();
  ctx.strokeStyle='#0f172a'; ctx.lineWidth=1.5; ctx.stroke();
}}

/* X-axis labels */
ctx.fillStyle='#64748b'; ctx.font='11px sans-serif'; ctx.textAlign='center';
const step=Math.max(1,Math.floor(n/15));
for (let i=0;i<n;i+=step) ctx.fillText(i, toX(i), H-pad.bottom+18);
ctx.fillText('Iteration', W/2, H-4);

/* Legend */
ctx.fillStyle='#7dd3fc'; ctx.beginPath(); ctx.arc(W-180,pad.top+10,5,0,Math.PI*2); ctx.fill();
ctx.fillStyle='#94a3b8'; ctx.font='11px sans-serif'; ctx.textAlign='left'; ctx.fillText('Initial',W-170,pad.top+14);
ctx.fillStyle='#c4b5fd'; ctx.beginPath(); ctx.arc(W-110,pad.top+10,5,0,Math.PI*2); ctx.fill();
ctx.fillText('BO',W-100,pad.top+14);
ctx.strokeStyle='#ef4444'; ctx.lineWidth=2;
ctx.beginPath(); ctx.moveTo(W-65,pad.top+10); ctx.lineTo(W-45,pad.top+10); ctx.stroke();
ctx.fillStyle='#94a3b8'; ctx.fillText('Best',W-40,pad.top+14);
</script>

</body>
</html>"""


# ── Dashboard server class ────────────────────────────────────────────────────

class CampaignDashboard:
    """
    Live dashboard with pause/stop/resume control.
    Served at http://localhost:<port>/ — open in any browser.
    """

    def __init__(self, port: int = DEFAULT_PORT, images_dir: str = "", dry_run: bool = False):
        self.port = port
        self._images_dir = images_dir
        self._dry_run = dry_run
        self._state = "running"
        self._data: dict = {}
        self._current_params: dict = {}
        self._lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        """Start the background HTTP server and open the dashboard in a browser."""
        db = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ('/', '/dashboard'):
                    with db._lock:
                        html = generate_html(db._data)
                    body = html.encode('utf-8')
                    self._ok('text/html; charset=utf-8', body)
                elif self.path.startswith('/camera-feed'):
                    img_path = db._get_latest_image()
                    if img_path:
                        try:
                            with open(img_path, 'rb') as f:
                                data = f.read()
                            ct = ('image/png' if str(img_path).lower().endswith('.png')
                                  else 'image/jpeg')
                            self.send_response(200)
                            self.send_header('Content-Type', ct)
                            self.send_header('Content-Length', str(len(data)))
                            self.send_header('Cache-Control', 'no-store, no-cache')
                            self.end_headers()
                            self.wfile.write(data)
                        except Exception:
                            self.send_response(404)
                            self.end_headers()
                    else:
                        self.send_response(404)
                        self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == '/control':
                    length = int(self.headers.get('Content-Length', 0))
                    try:
                        payload = json.loads(self.rfile.read(length))
                        action = payload.get('action', '')
                    except Exception:
                        action = ''

                    with db._lock:
                        prev = db._state
                        if action == 'pause' and db._state == 'running':
                            db._state = 'paused'
                        elif action == 'resume' and db._state == 'paused':
                            db._state = 'running'
                        elif action == 'stop' and db._state in ('running', 'paused', 'awaiting_confirmation'):
                            db._state = 'stopped'
                        elif action == 'stop_now' and db._state in ('running', 'paused', 'stop_now'):
                            db._state = 'stop_now'
                        elif action == 'confirm_next' and db._state == 'awaiting_confirmation':
                            db._state = 'running'
                        new = db._state

                    if prev != new:
                        print(f"\n[Dashboard] {prev.upper()} → {new.upper()}")

                    resp = json.dumps({'ok': True, 'state': new}).encode()
                    self._ok('application/json', resp)
                else:
                    self.send_response(404)
                    self.end_headers()

            def _ok(self, ct, body):
                self.send_response(200)
                self.send_header('Content-Type', ct)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass  # suppress server access logs

        import socket as _socket

        class _ReuseServer(ThreadingHTTPServer):
            allow_reuse_address = True

            def server_bind(self):
                self.socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
                super().server_bind()

        # Auto-increment port if the preferred one is busy
        port = self.port
        for attempt in range(10):
            try:
                self._server = _ReuseServer(('localhost', port), _Handler)
                self.port = port
                break
            except OSError:
                port += 1
        else:
            raise RuntimeError(f"Could not bind to any port in {self.port}–{port}")

        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()
        url = f"http://localhost:{self.port}/"
        webbrowser.open(url)
        print(f"[Dashboard] {url}")

    def stop(self):
        """Shut down the HTTP server."""
        if self._server:
            self._server.shutdown()

    # ── Camera feed ───────────────────────────────────────────────────────

    def _get_latest_image(self):
        """Return the Path of the most recently modified image in images_dir, or None."""
        from pathlib import Path
        if not self._images_dir:
            return None
        p = Path(self._images_dir)
        if not p.exists():
            return None
        imgs = [f for ext in ('jpg', 'jpeg', 'png') for f in p.rglob(f'*.{ext}')]
        return max(imgs, key=lambda f: f.stat().st_mtime) if imgs else None

    # ── State ─────────────────────────────────────────────────────────────

    def get_state(self) -> str:
        """Return current campaign state: running | paused | stopped | completed | awaiting_confirmation."""
        with self._lock:
            return self._state

    def set_state(self, state: str):
        with self._lock:
            self._state = state
            if 'status' in self._data:
                self._data['status']['state'] = state

    def set_current_params(self, params: dict):
        """Store the parameters currently being executed by the robot."""
        with self._lock:
            self._current_params = dict(params) if params else {}
            self._data["current_params"] = self._current_params

    def set_robot_status(self, step: str, ot2_status: str = "idle",
                         run_id: str = "", iteration: int = None):
        """Lightweight update — only refreshes the robot status fields.
        Call this before/after each sub-step for real-time feedback."""
        short_id = (run_id[:12] + "…") if len(run_id) > 12 else run_id
        with self._lock:
            self._data["robot_status"] = {
                "step":       step,
                "ot2_status": ot2_status,
                "run_id":     short_id,
                "iteration":  iteration,
            }

    # ── Data update ───────────────────────────────────────────────────────

    def update(
        self,
        optimizer,
        next_suggestion: Optional[dict] = None,
        stop_reason: Optional[str] = None,
        iter_results: Optional[list] = None,   # list of {protocol_file: str, ...}
    ):
        """
        Rebuild dashboard data from the current optimizer state.
        Call this after every iteration.
        """
        experiments = []
        best_val, bsf = -1, []
        for i in range(len(optimizer.y_observed)):
            best_val = max(best_val, optimizer.y_observed[i])
            bsf.append(best_val)
            proto = ""
            if iter_results and i < len(iter_results):
                proto = iter_results[i].get("protocol_file", "")
            experiments.append({
                "iteration":   i,
                "phase":       optimizer.phases[i],
                "params":      optimizer._raw_to_dict(optimizer.X_raw[i]),
                "accuracy":    optimizer.y_observed[i],
                "ei":          optimizer.ei_values[i],
                "best_so_far": bsf[i],
                "protocol_file": proto,
            })

        best_idx = (
            int(max(range(len(optimizer.y_observed)),
                    key=lambda i: optimizer.y_observed[i]))
            if optimizer.y_observed else None
        )

        # Decide new state from stop_reason
        if stop_reason:
            sl = stop_reason.lower()
            new_state = (
                "completed" if ("converged" in sl or "cap" in sl)
                else "stopped" if "stopped" in sl
                else None
            )
        else:
            new_state = None

        phase_label = (
            "Initial (LHS)" if optimizer.iteration < optimizer.n_initial
            else "Bayesian Opt"
        )

        pre_qc = [
            {"name": q.name, "passed": q.passed, "value": q.value,
             "rationale": getattr(q, 'rationale', ''), "timing": getattr(q, 'timing', '')}
            for q in optimizer.pre_experiment_qc
        ]

        with self._lock:
            current_state = new_state if new_state else self._state
            self._state = current_state
            self._data = {
                "experiments": experiments,
                "config": {
                    "max_iterations":    optimizer.max_iterations,
                    "n_initial":         optimizer.n_initial,
                    "convergence_tol":   optimizer.convergence_tol,
                    "convergence_window": optimizer.convergence_window,
                    "factors":           [f.name for f in optimizer.factors],
                },
                "status": {
                    "iteration":   optimizer.iteration,
                    "phase":       phase_label,
                    "state":       current_state,
                    "stop_reason": stop_reason or "",
                },
                "best": {
                    "accuracy":  optimizer.y_observed[best_idx] if best_idx is not None else 0,
                    "iteration": best_idx if best_idx is not None else 0,
                    "params":    optimizer._raw_to_dict(optimizer.X_raw[best_idx]) if best_idx is not None else {},
                },
                "next_suggestion":    next_suggestion,
                "current_params":     self._current_params,
                "qc_pre_experiment":  pre_qc,
                "dry_run":            self._dry_run,
            }
