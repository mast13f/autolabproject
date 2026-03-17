"""Campaign management dashboard — generates a live-updating HTML report.

Usage:
    from dashboard import CampaignDashboard
    dashboard = CampaignDashboard("campaign_dashboard.html")
    dashboard.update(optimizer)  # call after each iteration
"""

import json
import os
from typing import Optional


def generate_dashboard_html(data: dict) -> str:
    """Generate a complete self-contained HTML dashboard from optimizer state."""
    experiments = data["experiments"]
    config = data["config"]
    status = data["status"]
    best = data["best"]
    next_suggestion = data.get("next_suggestion")
    qc_pre = data.get("qc_pre_experiment", [])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="2">
<title>DOE Campaign Dashboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f172a; color: #e2e8f0; padding: 20px;
  }}
  .header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid #334155;
  }}
  .header h1 {{ font-size: 24px; color: #f8fafc; }}
  .header .status {{
    padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 600;
  }}
  .status-running {{ background: #065f46; color: #6ee7b7; }}
  .status-converged {{ background: #1e3a5f; color: #7dd3fc; }}
  .status-capped {{ background: #78350f; color: #fcd34d; }}

  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .card {{
    background: #1e293b; border-radius: 12px; padding: 20px;
    border: 1px solid #334155;
  }}
  .card .label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }}
  .card .value {{ font-size: 28px; font-weight: 700; color: #f8fafc; }}
  .card .sub {{ font-size: 12px; color: #64748b; margin-top: 4px; }}
  .card.highlight {{ border-color: #3b82f6; }}
  .card.highlight .value {{ color: #60a5fa; }}
  .card.best {{ border-color: #10b981; }}
  .card.best .value {{ color: #34d399; }}

  .panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
  @media (max-width: 900px) {{ .panels {{ grid-template-columns: 1fr; }} }}

  .panel {{
    background: #1e293b; border-radius: 12px; padding: 20px;
    border: 1px solid #334155;
  }}
  .panel h2 {{ font-size: 14px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 16px; }}

  .next-params {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .next-param {{
    background: #0f172a; border-radius: 8px; padding: 12px;
    border: 1px solid #334155;
  }}
  .next-param .name {{ font-size: 11px; color: #64748b; margin-bottom: 2px; }}
  .next-param .val {{ font-size: 20px; font-weight: 600; color: #fbbf24; }}

  .stop-criteria {{ list-style: none; }}
  .stop-criteria li {{
    padding: 10px 12px; margin-bottom: 8px; border-radius: 8px;
    background: #0f172a; border: 1px solid #334155; font-size: 13px;
  }}
  .stop-criteria .label-tag {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 600; margin-right: 8px;
  }}
  .tag-active {{ background: #065f46; color: #6ee7b7; }}
  .tag-triggered {{ background: #7c2d12; color: #fdba74; }}
  .tag-waiting {{ background: #1e293b; color: #64748b; }}

  .chart-container {{
    background: #1e293b; border-radius: 12px; padding: 20px;
    border: 1px solid #334155; margin-bottom: 24px;
  }}
  .chart-container h2 {{ font-size: 14px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 16px; }}
  canvas {{ width: 100% !important; height: 300px !important; }}

  .table-container {{
    background: #1e293b; border-radius: 12px; padding: 20px;
    border: 1px solid #334155; overflow-x: auto;
  }}
  .table-container h2 {{ font-size: 14px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  thead th {{
    text-align: left; padding: 10px 12px; color: #94a3b8; font-weight: 600;
    border-bottom: 2px solid #334155; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.5px; position: sticky; top: 0; background: #1e293b;
  }}
  tbody td {{ padding: 10px 12px; border-bottom: 1px solid #1e293b; }}
  tbody tr {{ background: #0f172a; }}
  tbody tr:hover {{ background: #1e293b; }}
  tbody tr.best-row {{ background: #052e16; }}
  tbody tr.best-row td {{ color: #34d399; font-weight: 600; }}
  .phase-badge {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 600;
  }}
  .phase-initial {{ background: #1e3a5f; color: #7dd3fc; }}
  .phase-bo {{ background: #3b1f6e; color: #c4b5fd; }}

  .bar-cell {{ position: relative; }}
  .bar-fill {{
    position: absolute; left: 0; top: 0; bottom: 0;
    opacity: 0.15; border-radius: 4px;
  }}

  .qc-section {{
    background: #1e293b; border-radius: 12px; padding: 20px;
    border: 1px solid #334155; margin-bottom: 24px;
  }}
  .qc-section h2 {{ font-size: 14px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 16px; }}
  .qc-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
  .qc-item {{
    background: #0f172a; border-radius: 8px; padding: 14px;
    border: 1px solid #334155; display: flex; align-items: flex-start; gap: 12px;
  }}
  .qc-icon {{
    width: 32px; height: 32px; border-radius: 8px; display: flex;
    align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0;
  }}
  .qc-pass {{ background: #065f46; }}
  .qc-fail {{ background: #7c2d12; }}
  .qc-item .qc-name {{ font-size: 13px; font-weight: 600; color: #e2e8f0; margin-bottom: 2px; }}
  .qc-item .qc-value {{ font-size: 12px; color: #94a3b8; }}
  .qc-item .qc-rationale {{ font-size: 11px; color: #64748b; margin-top: 4px; line-height: 1.4; }}
  .qc-item .qc-timing {{ font-size: 10px; color: #475569; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.3px; }}

  .qc-badge {{
    display: inline-block; padding: 1px 6px; border-radius: 3px;
    font-size: 10px; font-weight: 600;
  }}
  .qc-badge-pass {{ background: #065f46; color: #6ee7b7; }}
  .qc-badge-info {{ background: #1e3a5f; color: #7dd3fc; }}

  .footer {{
    text-align: center; padding: 16px; color: #475569; font-size: 12px;
    margin-top: 16px;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>DOE Campaign Dashboard</h1>
  <span class="status {_status_class(status['state'])}">{status['state'].upper()}</span>
</div>

<div class="cards">
  <div class="card highlight">
    <div class="label">Experiments Run</div>
    <div class="value">{status['iteration']}</div>
    <div class="sub">of {config['max_iterations']} max</div>
  </div>
  <div class="card best">
    <div class="label">Best Accuracy</div>
    <div class="value">{best['accuracy']:.1f}%</div>
    <div class="sub">Iteration #{best['iteration']}</div>
  </div>
  <div class="card">
    <div class="label">Current Phase</div>
    <div class="value" style="font-size:20px">{status['phase']}</div>
    <div class="sub">{config['n_initial']} initial + BO</div>
  </div>
  <div class="card">
    <div class="label">Mean Accuracy</div>
    <div class="value">{_mean_accuracy(experiments):.1f}%</div>
    <div class="sub">across all experiments</div>
  </div>
  <div class="card">
    <div class="label">Accuracy Std Dev</div>
    <div class="value">{_std_accuracy(experiments):.1f}</div>
    <div class="sub">variability measure</div>
  </div>
  <div class="card">
    <div class="label">Latest EI</div>
    <div class="value">{_latest_ei(experiments)}</div>
    <div class="sub">expected improvement</div>
  </div>
</div>

<div class="panels">
  <div class="panel">
    <h2>Next Suggested Parameters</h2>
    {_next_params_html(next_suggestion)}
  </div>
  <div class="panel">
    <h2>Stopping Criteria</h2>
    <ul class="stop-criteria">
      <li>
        <span class="label-tag {_criterion_tag(status, 'convergence')}">{_criterion_label(status, 'convergence')}</span>
        Accuracy improvement &lt; {config['convergence_tol']}% over {config['convergence_window']} BO iterations
      </li>
      <li>
        <span class="label-tag {_criterion_tag(status, 'cap')}">{_criterion_label(status, 'cap')}</span>
        Experiment cap at {config['max_iterations']} total experiments
      </li>
    </ul>
    <div style="margin-top:16px">
      <h2>Best Parameters Found</h2>
      {_best_params_html(best)}
    </div>
  </div>
</div>

{_qc_section_html(qc_pre, experiments)}

<div class="chart-container">
  <h2>Accuracy vs. Iteration</h2>
  <canvas id="chart"></canvas>
</div>

<div class="table-container">
  <h2>Experiment Log</h2>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Phase</th><th>Aspirate Speed</th><th>Dispense Speed</th>
        <th>Air Gap</th><th>Blow Out</th><th>Accuracy</th><th>EI</th><th>Best So Far</th>
        <th>Pre-Pipette Check</th><th>Post-Dispense Check</th>
      </tr>
    </thead>
    <tbody>
      {_experiments_rows(experiments, best['iteration'])}
    </tbody>
  </table>
</div>

<div class="footer">
  Auto-refreshes every 2 seconds &middot; Last updated: <span id="ts"></span>
  <script>document.getElementById('ts').textContent = new Date().toLocaleTimeString();</script>
</div>

<script>
const DATA = {json.dumps(_chart_data(experiments, config['n_initial']))};

const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
const dpr = window.devicePixelRatio || 1;
const rect = canvas.parentElement.getBoundingClientRect();
canvas.width = rect.width * dpr;
canvas.height = 300 * dpr;
canvas.style.width = rect.width + 'px';
canvas.style.height = '300px';
ctx.scale(dpr, dpr);

const W = rect.width, H = 300;
const pad = {{top:20, right:20, bottom:40, left:50}};
const pw = W - pad.left - pad.right, ph = H - pad.top - pad.bottom;

const n = DATA.accuracy.length;
const xScale = n > 1 ? pw / (n - 1) : pw / 2;
const yMin = 0, yMax = 100;

function toX(i) {{ return pad.left + i * xScale; }}
function toY(v) {{ return pad.top + ph - (v - yMin) / (yMax - yMin) * ph; }}

// Grid
ctx.strokeStyle = '#334155'; ctx.lineWidth = 0.5;
for (let v = 0; v <= 100; v += 20) {{
  ctx.beginPath(); ctx.moveTo(pad.left, toY(v)); ctx.lineTo(W - pad.right, toY(v)); ctx.stroke();
  ctx.fillStyle = '#64748b'; ctx.font = '11px sans-serif'; ctx.textAlign = 'right';
  ctx.fillText(v + '%', pad.left - 8, toY(v) + 4);
}}

// Phase boundary
if (DATA.n_initial > 0 && DATA.n_initial < n) {{
  const bx = toX(DATA.n_initial - 0.5);
  ctx.strokeStyle = '#475569'; ctx.lineWidth = 1; ctx.setLineDash([5, 5]);
  ctx.beginPath(); ctx.moveTo(bx, pad.top); ctx.lineTo(bx, H - pad.bottom); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#64748b'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('Initial | BO', bx, pad.top - 5);
}}

// Best-so-far line
ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 2;
ctx.beginPath();
for (let i = 0; i < n; i++) {{
  const x = toX(i), y = toY(DATA.best_so_far[i]);
  i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
}}
ctx.stroke();

// Scatter points
for (let i = 0; i < n; i++) {{
  const x = toX(i), y = toY(DATA.accuracy[i]);
  ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2);
  ctx.fillStyle = DATA.colors[i]; ctx.fill();
  ctx.strokeStyle = '#0f172a'; ctx.lineWidth = 1.5; ctx.stroke();
}}

// X axis labels
ctx.fillStyle = '#64748b'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
const step = Math.max(1, Math.floor(n / 15));
for (let i = 0; i < n; i += step) {{
  ctx.fillText(i, toX(i), H - pad.bottom + 20);
}}
ctx.fillText('Iteration', W / 2, H - 5);

// Legend
ctx.fillStyle = '#7dd3fc'; ctx.beginPath(); ctx.arc(W - 200, pad.top + 10, 5, 0, Math.PI*2); ctx.fill();
ctx.fillStyle = '#94a3b8'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
ctx.fillText('Initial', W - 190, pad.top + 14);
ctx.fillStyle = '#c4b5fd'; ctx.beginPath(); ctx.arc(W - 130, pad.top + 10, 5, 0, Math.PI*2); ctx.fill();
ctx.fillStyle = '#94a3b8'; ctx.fillText('BO', W - 120, pad.top + 14);
ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 2;
ctx.beginPath(); ctx.moveTo(W - 80, pad.top + 10); ctx.lineTo(W - 55, pad.top + 10); ctx.stroke();
ctx.fillStyle = '#94a3b8'; ctx.fillText('Best', W - 50, pad.top + 14);
</script>

</body>
</html>"""


def _status_class(state: str) -> str:
    if state == "running":
        return "status-running"
    elif state == "converged":
        return "status-converged"
    return "status-capped"


def _mean_accuracy(experiments: list) -> float:
    if not experiments:
        return 0.0
    return sum(e["accuracy"] for e in experiments) / len(experiments)


def _std_accuracy(experiments: list) -> float:
    if len(experiments) < 2:
        return 0.0
    import statistics
    return statistics.stdev(e["accuracy"] for e in experiments)


def _latest_ei(experiments: list) -> str:
    for e in reversed(experiments):
        if e["ei"] is not None:
            return f"{e['ei']:.4f}"
    return "N/A"


def _next_params_html(suggestion: Optional[dict]) -> str:
    if not suggestion:
        return '<div style="color:#64748b;padding:20px;text-align:center">Waiting for first BO iteration...</div>'
    html = '<div class="next-params">'
    for name, val in suggestion.items():
        html += f'<div class="next-param"><div class="name">{name}</div><div class="val">{val}</div></div>'
    html += '</div>'
    return html


def _best_params_html(best: dict) -> str:
    html = '<div class="next-params" style="margin-top:8px">'
    for name, val in best["params"].items():
        html += f'<div class="next-param"><div class="name">{name}</div><div class="val" style="color:#34d399">{val}</div></div>'
    html += '</div>'
    return html


def _criterion_tag(status: dict, criterion: str) -> str:
    if status["state"] == "running":
        return "tag-active"
    if criterion == "convergence" and status.get("stop_reason", "").startswith("Accuracy converged"):
        return "tag-triggered"
    if criterion == "cap" and status.get("stop_reason", "").startswith("Experiment cap"):
        return "tag-triggered"
    return "tag-waiting"


def _criterion_label(status: dict, criterion: str) -> str:
    if status["state"] == "running":
        return "WATCHING"
    if criterion == "convergence" and status.get("stop_reason", "").startswith("Accuracy converged"):
        return "TRIGGERED"
    if criterion == "cap" and status.get("stop_reason", "").startswith("Experiment cap"):
        return "TRIGGERED"
    return "NOT MET"


def _qc_section_html(qc_pre: list, experiments: list) -> str:
    """Generate the QC metrics section with pre-experiment checks and per-iteration summary."""
    html = '<div class="qc-section">\n'
    html += '  <h2>Quality Control Metrics</h2>\n'

    # Pre-experiment checks
    if qc_pre:
        html += '  <h3 style="font-size:12px;color:#64748b;margin-bottom:12px;text-transform:uppercase;letter-spacing:0.5px">Pre-Experiment Checks</h3>\n'
        html += '  <div class="qc-grid">\n'
        for qc in qc_pre:
            icon_class = "qc-pass" if qc["passed"] else "qc-fail"
            icon = "&#10003;" if qc["passed"] else "&#10007;"
            html += f'''    <div class="qc-item">
      <div class="qc-icon {icon_class}">{icon}</div>
      <div>
        <div class="qc-name">{qc["name"]}</div>
        <div class="qc-value">{qc["value"]}</div>
        <div class="qc-rationale">{qc.get("rationale", "")}</div>
        <div class="qc-timing">{qc.get("timing", "")}</div>
      </div>
    </div>\n'''
        html += '  </div>\n'

    # Per-iteration QC summary
    if experiments and any(e.get("qc") for e in experiments):
        html += '  <h3 style="font-size:12px;color:#64748b;margin:16px 0 12px;text-transform:uppercase;letter-spacing:0.5px">Per-Iteration Checks (Latest)</h3>\n'
        # Show the latest iteration's QC
        latest = None
        for e in reversed(experiments):
            if e.get("qc"):
                latest = e
                break
        if latest:
            html += '  <div class="qc-grid">\n'
            for qc in latest["qc"]:
                icon_class = "qc-pass" if qc["passed"] else "qc-fail"
                icon = "&#10003;" if qc["passed"] else "&#10007;"
                html += f'''    <div class="qc-item">
      <div class="qc-icon {icon_class}">{icon}</div>
      <div>
        <div class="qc-name">{qc["name"]} <span style="color:#475569;font-size:11px">(Iter #{latest["iteration"]})</span></div>
        <div class="qc-value">{qc["value"]}</div>
      </div>
    </div>\n'''
            html += '  </div>\n'

    html += '</div>'
    return html


def _experiments_rows(experiments: list, best_iter: int) -> str:
    rows = ""
    for e in experiments:
        is_best = e["iteration"] == best_iter
        row_class = ' class="best-row"' if is_best else ""
        phase_class = "phase-initial" if e["phase"] == "INITIAL" else "phase-bo"
        ei_str = f"{e['ei']:.4f}" if e["ei"] is not None else "-"
        bar_width = e["accuracy"]
        bar_color = "#34d399" if is_best else ("#7dd3fc" if e["phase"] == "INITIAL" else "#c4b5fd")

        # QC columns
        qc_list = e.get("qc", [])
        pre_pipette = "-"
        post_dispense = "-"
        for qc in qc_list:
            if "Before" in qc["name"]:
                badge_cls = "qc-badge-pass" if "No liquid" in qc["value"] else "qc-badge-info"
                pre_pipette = f'<span class="qc-badge {badge_cls}">{qc["value"]}</span>'
            elif "After" in qc["name"]:
                badge_cls = "qc-badge-pass" if "No liquid" in qc["value"] else "qc-badge-info"
                post_dispense = f'<span class="qc-badge {badge_cls}">{qc["value"]}</span>'

        rows += f"""<tr{row_class}>
          <td>{e['iteration']}</td>
          <td><span class="phase-badge {phase_class}">{e['phase']}</span></td>
          <td>{e['params']['aspirate_speed']}</td>
          <td>{e['params']['dispense_speed']}</td>
          <td>{e['params']['air_gap']}</td>
          <td>{e['params']['blow_out']}</td>
          <td class="bar-cell">
            <div class="bar-fill" style="width:{bar_width}%;background:{bar_color}"></div>
            {e['accuracy']:.2f}
          </td>
          <td>{ei_str}</td>
          <td>{e['best_so_far']:.2f}</td>
          <td>{pre_pipette}</td>
          <td>{post_dispense}</td>
        </tr>\n"""
    return rows


def _chart_data(experiments: list, n_initial: int) -> dict:
    accuracies = [e["accuracy"] for e in experiments]
    best_so_far = []
    best = -1
    for a in accuracies:
        best = max(best, a)
        best_so_far.append(best)
    colors = ["#7dd3fc" if e["phase"] == "INITIAL" else "#c4b5fd" for e in experiments]
    return {
        "accuracy": accuracies,
        "best_so_far": best_so_far,
        "colors": colors,
        "n_initial": n_initial,
    }


class CampaignDashboard:
    """Generates and updates an HTML dashboard file from DOEOptimizer state."""

    def __init__(self, html_path: str = "campaign_dashboard.html"):
        self.html_path = html_path

    def update(self, optimizer, next_suggestion: Optional[dict] = None,
               stop_reason: Optional[str] = None):
        """Re-generate the HTML dashboard from current optimizer state."""
        experiments = []
        best_val = -1
        best_so_far_list = []
        for i in range(len(optimizer.y_observed)):
            best_val = max(best_val, optimizer.y_observed[i])
            best_so_far_list.append(best_val)
            # Per-iteration QC
            iter_qc = []
            if i < len(optimizer.per_iteration_qc):
                iter_qc = [{"name": q.name, "passed": q.passed, "value": q.value}
                           for q in optimizer.per_iteration_qc[i]]
            experiments.append({
                "iteration": i,
                "phase": optimizer.phases[i],
                "params": optimizer._raw_to_dict(optimizer.X_raw[i]),
                "accuracy": optimizer.y_observed[i],
                "ei": optimizer.ei_values[i],
                "best_so_far": best_so_far_list[i],
                "qc": iter_qc,
            })

        if optimizer.y_observed:
            best_idx = int(max(range(len(optimizer.y_observed)),
                               key=lambda i: optimizer.y_observed[i]))
        else:
            best_idx = None

        if stop_reason and "converged" in stop_reason.lower():
            state = "converged"
        elif stop_reason and "cap" in stop_reason.lower():
            state = "capped"
        else:
            state = "running"

        phase = "Initial (LHS)" if optimizer.iteration < optimizer.n_initial else "Bayesian Opt"

        # Pre-experiment QC
        pre_qc = [{"name": q.name, "passed": q.passed, "value": q.value,
                    "rationale": q.rationale, "timing": q.timing}
                   for q in optimizer.pre_experiment_qc]

        data = {
            "experiments": experiments,
            "config": {
                "max_iterations": optimizer.max_iterations,
                "n_initial": optimizer.n_initial,
                "convergence_tol": optimizer.convergence_tol,
                "convergence_window": optimizer.convergence_window,
            },
            "status": {
                "iteration": optimizer.iteration,
                "phase": phase,
                "state": state,
                "stop_reason": stop_reason or "",
            },
            "best": {
                "accuracy": optimizer.y_observed[best_idx] if best_idx is not None else 0,
                "iteration": best_idx if best_idx is not None else 0,
                "params": optimizer._raw_to_dict(optimizer.X_raw[best_idx]) if best_idx is not None else {},
            },
            "next_suggestion": next_suggestion,
            "qc_pre_experiment": pre_qc,
        }

        html = generate_dashboard_html(data)
        with open(self.html_path, "w") as f:
            f.write(html)
