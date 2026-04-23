"""
AutoLab Experiment Runner
==========================
Closed-loop optimization pipeline:

  Bayesian Optimizer
       ↓  suggested parameters
  Protocol Generator  →  saves iter_NNN_*.py to protocols/
       ↓  protocol file
  OT-2 Controller     →  runs on robot
       ↓  images saved to captured_images/
  Image Subtraction   →  saves results to results/iter_NNN/
       ↓  dispense score
  Bayesian Optimizer  ←  tell(params, score)

Repeats until stopping criteria met or user presses Stop in dashboard.

Usage:
    python -m autolab.runner              # real robot run
    python -m autolab.runner --dry-run    # no robot, random score (for testing)
"""

import csv as _csv
import shutil
import sys
import time
from pathlib import Path


# ── Results CSV ───────────────────────────────────────────────────────────────

def _write_results_csv(csv_path: Path, iteration: int, phase: str,
                        params: dict, score: float, ei_val,
                        best_so_far: float, protocol_file: str):
    """Append one result row to the campaign CSV, writing headers on first call."""
    write_header = not csv_path.exists()
    factor_names = list(params.keys())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        if write_header:
            w.writerow(
                ["iteration", "timestamp", "phase"]
                + factor_names
                + ["score", "ei", "best_so_far", "protocol_file"]
            )
        w.writerow(
            [iteration, time.strftime("%Y-%m-%d %H:%M:%S"), phase]
            + [params[n] for n in factor_names]
            + [
                f"{score:.2f}",
                f"{ei_val:.6f}" if ei_val is not None else "",
                f"{best_so_far:.2f}",
                protocol_file,
            ]
        )

# Set non-interactive matplotlib backend BEFORE any import that loads matplotlib.pyplot
# This prevents plot_convergence() from opening a pop-up window
import matplotlib
matplotlib.use('Agg')

# ── Path setup — add project root so all packages resolve ─────────────────────
ROOT = Path(__file__).parent.parent   # autolab/ → autolabproject/
sys.path.insert(0, str(ROOT))

import autolab.config as cfg
from autolab.dashboard import CampaignDashboard
from autolab.protocol_gen import generate_protocol
from autolab.wizard import SetupWizard, build_factors_meta
from optimizer import PipettingDOEOptimizer, simulate_qc_checks_pre_experiment, simulate_qc_checks_per_iteration


# ── Image analysis (new subtraction pipeline) ────────────────────────────────

def run_analysis(image_dir: Path, output_dir: Path,
                 threshold: int = 12) -> tuple:
    """Run image-subtraction analysis on *image_dir*.

    Returns ``(score, px_data)`` where *score* is 0-100 and *px_data* is a
    dict with pixel diagnostics (e.g. ``{"after_dispense": 1234}``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from autolab.image_analysis import measure_from_images, aggregate_score
        import numpy as np

        col_results = measure_from_images(
            str(image_dir), str(output_dir), threshold=threshold,
        )
        score = aggregate_score(col_results)
        if score is None:
            print("    [WARN] No scoreable columns found.")
            return 0.0, {}

        px_vals = [
            v["px_after_dispense"]
            for v in col_results.values()
            if v.get("px_after_dispense") is not None
        ]
        px = {"after_dispense": int(np.mean(px_vals))} if px_vals else {}

        print(f"    Analysis: score {score:.1f}/100")
        return score, px
    except Exception as e:
        print(f"    [WARN] Analysis failed: {e}")
        return 0.0, {}


# ── Image collection ──────────────────────────────────────────────────────────

def collect_new_images(images_dir: Path, since: float) -> list:
    """Return image files created at or after `since` (Unix timestamp)."""
    if not images_dir.exists():
        return []
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for f in images_dir.glob(ext):
            try:
                if f.stat().st_mtime >= since - 2.0:   # 2 s tolerance
                    imgs.append(f)
            except OSError:
                pass
    return sorted(imgs)


# ── Camera server auto-launch ─────────────────────────────────────────────────

def _camera_server_alive(ip: str, port: int, timeout: float = 2.0) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://{ip}:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _start_camera_server(camera_ip: str, camera_port: int,
                          camera_index: int, save_dir: Path):
    """Launch camera/server.py as a subprocess if it isn't already running."""
    import subprocess
    if _camera_server_alive("127.0.0.1", camera_port):
        print(f"[CAMERA] Server already running on port {camera_port} — reusing.")
        return None
    server_script = ROOT / "camera" / "server.py"
    if not server_script.exists():
        print(f"[CAMERA] server.py not found at {server_script} — skipping auto-launch.")
        return None
    cmd = [sys.executable, str(server_script),
           "--camera-index", str(camera_index),
           "--port",         str(camera_port),
           "--save-dir",     str(save_dir)]
    print(f"[CAMERA] Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    for _ in range(16):
        time.sleep(0.5)
        if _camera_server_alive("127.0.0.1", camera_port):
            print(f"[CAMERA] Camera server ready on port {camera_port}.")
            return proc
    print("[CAMERA] WARNING: Camera server did not respond in time — continuing anyway.")
    return proc


# ── Pause helper ─────────────────────────────────────────────────────────────

def _handle_pause(dashboard) -> str | None:
    """Check dashboard state and block while paused.

    Returns ``None`` to continue, or a stop reason string if the user
    stopped the campaign while it was paused.
    """
    state = dashboard.get_state()
    if state == "paused":
        print("  [PAUSED] Waiting for Resume in dashboard…")
        while True:
            time.sleep(2)
            s = dashboard.get_state()
            if s in ("stopped", "stop_now"):
                return "Campaign stopped by user"
            if s == "running":
                print("  [RESUMED]")
                return None
    if state in ("stopped", "stop_now"):
        return "Campaign stopped by user"
    return None


# ── Interruptible OT-2 run ────────────────────────────────────────────────────

def run_ot2_interruptible(robot, protocol_path: Path, run_time_params: dict,
                           dashboard, iteration: int,
                           labware_offsets: list | None = None) -> str:
    """
    Upload and start a protocol, polling status every 5 s.
    Checks dashboard state on every poll:
      - "paused"   → pauses the OT-2 and waits for resume
      - "stop_now" → immediately stops the robot and returns "stopped_by_user"
    Returns the final OT-2 run status string.
    """
    terminal = {"succeeded", "failed", "stopped"}

    print(f"  Uploading protocol…")
    dashboard.set_robot_status("Uploading protocol", "uploading", "", iteration)
    protocol_id = robot.upload_protocol(str(protocol_path))

    print(f"  Creating run…")
    run_id = robot.create_run(protocol_id, run_time_params,
                               labware_offsets=labware_offsets)
    robot.start_run(run_id)
    dashboard.set_robot_status(f"OT-2 running — Iter {iteration:03d}", "running", run_id, iteration)

    last_status = None
    while True:
        # ── Check for pause / immediate stop ─────────────────────────
        ds = dashboard.get_state()
        if ds == "stop_now":
            print("\n  [STOP NOW] Halting robot immediately…")
            robot.stop_run(run_id)
            dashboard.set_robot_status("Stopped by user", "stopped", run_id, iteration)
            return "stopped_by_user"
        if ds == "paused":
            print("  [PAUSED] Pausing OT-2…")
            try:
                robot.pause_run(run_id)
            except Exception:
                pass  # robot may not support pause, or run already idle
            dashboard.set_robot_status(
                f"OT-2 paused — Iter {iteration:03d}", "paused", run_id, iteration
            )
            while True:
                time.sleep(2)
                s = dashboard.get_state()
                if s == "stop_now":
                    print("\n  [STOP NOW] Halting robot immediately…")
                    robot.stop_run(run_id)
                    dashboard.set_robot_status("Stopped by user", "stopped", run_id, iteration)
                    return "stopped_by_user"
                if s in ("stopped",):
                    print("\n  [STOP] Stopping after current action…")
                    robot.stop_run(run_id)
                    dashboard.set_robot_status("Stopped by user", "stopped", run_id, iteration)
                    return "stopped_by_user"
                if s == "running":
                    print("  [RESUMED] Resuming OT-2…")
                    try:
                        robot.resume_run(run_id)
                    except Exception:
                        # Fallback: re-issue play action
                        pass
                    dashboard.set_robot_status(
                        f"OT-2 running — Iter {iteration:03d}", "running", run_id, iteration
                    )
                    break

        run = robot.get_run(run_id)
        status = run.get("status", "unknown")

        if status != last_status:
            ts = time.strftime("%H:%M:%S")
            print(f"  [{ts}] OT-2: {status}")
            dashboard.set_robot_status(
                f"OT-2 running — Iter {iteration:03d}", status, run_id, iteration
            )
            last_status = status

        if status in terminal:
            errors = robot.get_run_errors(run_id)
            if errors:
                for e in errors:
                    print(f"  [OT-2 ERROR] {e.get('errorType','?')}: {e.get('detail','')}")
            return status

        time.sleep(5)


# ── Main campaign loop ────────────────────────────────────────────────────────

def run_campaign(dry_run: bool = False, wizard_config: dict | None = None):
    """
    Run the full optimization campaign.

    Args:
        dry_run:       Skip OT-2 and use random scores.
        wizard_config: Dict from SetupWizard.wait_for_start(), overrides cfg defaults
                       when provided. Keys: dry_run, lights, active_factors,
                       fixed_params, initial_mode, n_initial, custom_initial,
                       max_iterations, convergence_window, convergence_tol,
                       robot_ip, camera_ip, camera_port, camera_index.
    """
    import random

    # ── Apply wizard config ──────────────────────────────────────────────
    wc = wizard_config or {}
    if wc.get("dry_run") is not None:
        dry_run = wc["dry_run"]

    # Filter factors to only active ones (wizard may have unselected some)
    active_names = wc.get("active_factors")
    factors = [f for f in cfg.FACTORS if active_names is None or f.name in active_names] or cfg.FACTORS
    fixed_params = wc.get("fixed_params", {})

    n_initial         = wc.get("n_initial",          cfg.N_INITIAL)
    max_iterations    = wc.get("max_iterations",      cfg.MAX_ITERATIONS)
    convergence_tol   = wc.get("convergence_tol",     cfg.CONVERGENCE_TOL)
    convergence_window= wc.get("convergence_window",  cfg.CONVERGENCE_WINDOW)
    lights_on         = wc.get("lights", True)
    custom_initial    = wc.get("custom_initial", [])
    initial_mode      = wc.get("initial_mode", "algorithm")

    # Hardware settings from wizard (override cfg defaults when provided)
    camera_ip    = wc.get("camera_ip")     or cfg.CAMERA_SERVER_IP
    camera_port  = int(wc.get("camera_port") or cfg.CAMERA_SERVER_PORT)
    camera_index = int(wc.get("camera_index") if wc.get("camera_index") is not None
                       else cfg.CAMERA_INDEX)
    robot_ip     = wc.get("robot_ip")     or cfg.ROBOT_IP

    # Image analysis threshold
    image_threshold = getattr(cfg, "IMAGE_THRESHOLD", 12)

    print("=" * 70)
    print("AutoLab Experiment Campaign")
    print(f"  Mode     : {'DRY RUN (no robot, random scores)' if dry_run else 'LIVE'}")
    print(f"  Factors  : {[f.name for f in factors]}")
    if fixed_params:
        print(f"  Fixed    : {fixed_params}")
    print(f"  Liquid   : {cfg.LIQUID_TYPE} @ {cfg.CONCENTRATION_PCT}%")
    print(f"  Initial  : {n_initial} {'custom' if initial_mode == 'custom' else 'LHS'} points")
    print(f"  Max      : {max_iterations} experiments")
    print(f"  Conv.    : <{convergence_tol}% improvement over {convergence_window} BO iters")
    print(f"  Dashboard: http://localhost:{cfg.DASHBOARD_PORT}/")
    print("=" * 70)

    # ── Directories ──────────────────────────────────────────────────────
    cfg.PROTOCOLS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = PipettingDOEOptimizer(
        liquid_type=cfg.LIQUID_TYPE,
        concentration_pct=cfg.CONCENTRATION_PCT,
        factors=factors,
        n_initial=n_initial,
        max_iterations=max_iterations,
        xi=cfg.XI,
        convergence_tol=convergence_tol,
        convergence_window=convergence_window,
        image_threshold=image_threshold,
        training_csv=getattr(cfg, "TRAINING_CSV", "training_data.csv"),
        random_seed=cfg.RANDOM_SEED,
        csv_path=str(cfg.RESULTS_DIR / "campaign_log.csv"),
    )

    # Inject custom initial experiments if user provided them
    if initial_mode == "custom" and custom_initial:
        optimizer._custom_initial = custom_initial

    # ── Dashboard ─────────────────────────────────────────────────────────
    dashboard = CampaignDashboard(
        port=cfg.DASHBOARD_PORT,
        images_dir=str(cfg.IMAGES_DIR),
        dry_run=dry_run,
    )
    dashboard.start()
    dashboard.update(optimizer)

    # ── Robot + camera server ─────────────────────────────────────────────
    robot    = None
    cam_proc = None
    if not dry_run:
        from robot.controller import OT2
        try:
            robot = OT2(ip=robot_ip)
            robot.lights(lights_on)
        except ConnectionError as e:
            print(f"\n[ERROR] {e}")
            print("  Tip: run with --dry-run to test without a robot.")
            dashboard.stop()
            sys.exit(1)
        cam_proc = _start_camera_server(camera_ip, camera_port, camera_index, cfg.IMAGES_DIR)

    # ── Pre-experiment QC ─────────────────────────────────────────────────
    print("\n--- Pre-Experiment QC ---")
    optimizer.pre_experiment_qc = simulate_qc_checks_pre_experiment()
    for qc in optimizer.pre_experiment_qc:
        marker = "PASS" if qc.passed else "FAIL"
        print(f"  [{marker}] {qc.name}: {qc.value}")
    dashboard.set_robot_status("Pre-experiment QC complete — ready", "idle")
    dashboard.update(optimizer)

    # ── Per-iteration record (for dashboard) ──────────────────────────────
    iter_records: list = []  # [{protocol_file, images_dir, analysis_dir}, ...]
    stop_reason = None

    # Tip-rack column tracking — incremented per run, reset to 1 only when the
    # user acknowledges a physical rack swap on the confirmation banner.
    tip_column_next = 1
    dashboard.set_tip_rack_status(next_column=tip_column_next, exhausted=False)

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:

        # ── Control state check (pause / stop) ────────────────────────
        stop_reason = _handle_pause(dashboard)
        if stop_reason:
            print(f"\n[STOP] {stop_reason}")
            break

        # ── Hard cap check ────────────────────────────────────────────
        if optimizer.iteration >= max_iterations:
            stop_reason = (
                f"Experiment cap reached: {max_iterations} experiments "
                f"completed without convergence"
            )
            print(f"\n*** {stop_reason} ***")
            break

        # ── Convergence check (only meaningful in BO phase) ───────────
        converged, conv_reason = optimizer.check_convergence()
        if converged:
            stop_reason = conv_reason
            print(f"\n*** CONVERGED: {stop_reason} ***")
            break

        # ── Get next parameters ───────────────────────────────────────
        n = optimizer.iteration

        # Use custom initial point if available
        custom_pool = getattr(optimizer, "_custom_initial", [])
        if custom_pool and n < len(custom_pool):
            params = dict(custom_pool[n])
            ei_val = None
        else:
            try:
                params, ei_val = optimizer.suggest()
            except Exception as e:
                print(f"  [WARN] Optimizer suggestion failed ({e}), using random point")
                params = {}
                for f in factors:
                    if f.kind == "categorical":
                        params[f.name] = int(optimizer.rng.choice(f.levels))
                    else:
                        params[f.name] = round(float(optimizer.rng.uniform(f.low, f.high)), 2)
                ei_val = None

        # Merge in fixed parameters (not optimized)
        params.update(fixed_params)

        phase_label = "INITIAL" if n < n_initial else "BO"

        print(f"\n{'='*60}")
        print(f"Iteration {n:03d} | Phase: {phase_label}")
        for k, v in params.items():
            print(f"  {k} = {v}")
        if ei_val is not None:
            print(f"  EI = {ei_val:.4f}")

        # Show upcoming params in dashboard while experiment runs
        dashboard.set_robot_status(f"Planning Iter {n:03d} — generating protocol", "idle", "", n)
        dashboard.update(optimizer, next_suggestion=params, iter_results=iter_records)

        # ── Generate protocol file ────────────────────────────────────
        # Use the tracked tip-rack column — advances monotonically; only reset
        # to 1 when the user acknowledges a physical rack swap.
        tip_column = tip_column_next
        print(f"\n  Generating protocol file… (tip column A{tip_column})")
        protocol_path = generate_protocol(
            params, n,
            protocols_dir=cfg.PROTOCOLS_DIR,
            num_columns=cfg.NUM_COLUMNS,
            transfer_volume=cfg.TRANSFER_VOLUME,
            camera_ip=camera_ip,
            camera_port=camera_port,
            settle_seconds=cfg.SETTLE_SECONDS,
            camera_height_mm=cfg.CAMERA_HEIGHT_MM,
            tip_column=tip_column,
        )

        # ── Pause check before experiment ────────────────────────────
        stop_reason = _handle_pause(dashboard)
        if stop_reason:
            print(f"\n[STOP] {stop_reason}")
            break

        # ── Run experiment ────────────────────────────────────────────
        iter_dir = cfg.RESULTS_DIR / f"iter_{n:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        score = 0.0
        px_data: dict = {}

        # Show parameters being executed now
        dashboard.set_current_params(params)

        if dry_run:
            dashboard.set_robot_status(f"DRY RUN — Iter {n:03d} simulating", "running", "", n)
            print("  [DRY RUN] Skipping OT-2 — generating random score")
            time.sleep(0.5)
            score = round(random.uniform(40, 99), 2)
            px_data = {}
            print(f"  Simulated score: {score:.1f}")
            dashboard.set_robot_status(f"DRY RUN — Iter {n:03d} complete ({score:.1f})", "succeeded", "", n)

        else:
            # Use wizard-selected camera IP/port for OT-2 runtime params
            run_time_params = {
                "camera_server_ip":   camera_ip,
                "camera_server_port": camera_port,
                "capture_enabled":    True,
                "settle_seconds":     cfg.SETTLE_SECONDS,
                "camera_height_mm":   cfg.CAMERA_HEIGHT_MM,
            }

            print(f"\n  Running on OT-2…")
            run_start = time.time()
            ot2_result = "failed"
            try:
                ot2_result = run_ot2_interruptible(
                    robot, protocol_path, run_time_params, dashboard, n,
                    labware_offsets=cfg.LABWARE_OFFSETS,
                )
            except Exception as e:
                print(f"  [ERROR] OT-2 run failed: {e}")
                dashboard.set_robot_status(f"OT-2 error — Iter {n:03d}", "failed", "", n)

            if ot2_result == "stopped_by_user":
                stop_reason = "Campaign stopped by user (Stop Now)"
                # Still record the iteration with score=0
                optimizer.per_iteration_qc.append(simulate_qc_checks_per_iteration(0))
                optimizer.tell(params, 0.0, ei_val)
                record = {"protocol_file": protocol_path.name}
                iter_records.append(record)
                _write_results_csv(
                    csv_path=cfg.RESULTS_DIR / "results.csv",
                    iteration=n, phase=optimizer.phases[-1], params=params,
                    score=0.0, ei_val=ei_val,
                    best_so_far=max(optimizer.y_observed) if optimizer.y_observed else 0.0,
                    protocol_file=protocol_path.name,
                )
                dashboard.update(optimizer, stop_reason=stop_reason, iter_results=iter_records)
                break

            if ot2_result == "succeeded":
                # ── Collect images ────────────────────────────────────
                dashboard.set_robot_status(f"Collecting images — Iter {n:03d}", "succeeded", "", n)
                print("\n  Collecting images…")
                new_imgs = collect_new_images(cfg.IMAGES_DIR, run_start)
                image_dir = iter_dir / "images"
                image_dir.mkdir(exist_ok=True)
                if new_imgs:
                    for img in new_imgs:
                        shutil.copy2(img, image_dir / img.name)
                    print(f"  {len(new_imgs)} images → {image_dir.relative_to(ROOT)}")
                else:
                    print("  [WARN] No new images found — check camera server")

                # ── Image subtraction analysis ───────────────────────
                dashboard.set_robot_status(f"Analysing images — Iter {n:03d}", "succeeded", "", n)
                print("\n  Running image subtraction analysis…")
                analysis_dir = iter_dir / "analysis"
                score, px_data = run_analysis(image_dir, analysis_dir, image_threshold)
            else:
                score = 0.0
                px_data = {}

        # ── Update optimizer ──────────────────────────────────────────
        optimizer.per_iteration_qc.append(
            simulate_qc_checks_per_iteration(score)
        )
        optimizer.tell(params, score, ei_val, px=px_data)

        # ── Record for dashboard ──────────────────────────────────────
        record = {"protocol_file": protocol_path.name}
        if not dry_run:
            record["images_dir"]   = str(iter_dir / "images")
            record["analysis_dir"] = str(iter_dir / "analysis")
        iter_records.append(record)

        # ── Write results CSV ─────────────────────────────────────────
        best_so_far = max(optimizer.y_observed)
        _write_results_csv(
            csv_path=cfg.RESULTS_DIR / "results.csv",
            iteration=n,
            phase=optimizer.phases[-1],
            params=params,
            score=score,
            ei_val=ei_val,
            best_so_far=best_so_far,
            protocol_file=protocol_path.name,
        )

        # ── Advance tip-rack column ───────────────────────────────────
        # This iteration consumed column `tip_column`. If that was A12, the
        # rack is now exhausted — next iteration cannot run until the user
        # swaps racks via the dashboard.
        tip_column_next = tip_column + 1
        rack_exhausted  = tip_column_next > 12
        dashboard.set_tip_rack_status(
            next_column=(1 if rack_exhausted else tip_column_next),
            exhausted=rack_exhausted,
        )
        if rack_exhausted:
            print(f"  [RACK] Column A12 used — tip rack is now EXHAUSTED. "
                  f"Replace before continuing.")

        # ── Dashboard update ──────────────────────────────────────────
        dashboard.set_robot_status(f"Iter {n:03d} complete — score {score:.1f}", "idle", "", n)
        dashboard.update(optimizer, iter_results=iter_records)
        print(f"\n  Best so far: {best_so_far:.1f}")

        # ── Post-tell convergence check (catches convergence immediately) ─
        if optimizer.phases[-1] == "BO":
            converged, conv_reason = optimizer.check_convergence()
            if converged:
                stop_reason = conv_reason
                print(f"\n*** CONVERGED: {stop_reason} ***")
                break

        # ── Hard cap check (avoid waiting for confirmation when cap reached) ─
        if optimizer.iteration >= max_iterations:
            stop_reason = (
                f"Experiment cap reached: {max_iterations} experiments "
                f"completed without convergence"
            )
            print(f"\n*** {stop_reason} ***")
            break

        # ── Await user confirmation before next experiment ────────────
        # Compute next suggestion preview so the user can see what will run
        try:
            preview_params, _ = optimizer.suggest()
            preview_params.update(fixed_params)
        except Exception:
            preview_params = None
        dashboard.update(optimizer, next_suggestion=preview_params, iter_results=iter_records)
        dashboard.set_state("awaiting_confirmation")
        dashboard.set_robot_status(
            "Waiting for user confirmation — replace labware / refill if needed",
            "idle", "", n,
        )
        print("\n  [WAITING] Awaiting user confirmation in dashboard before next experiment…")

        while True:
            time.sleep(2)
            s = dashboard.get_state()
            if s == "running":
                ack = dashboard.consume_rack_replaced_ack()
                if rack_exhausted:
                    # Rack was exhausted — only accept progression via the
                    # explicit "Rack Replaced" ack. Any other confirmation
                    # is ignored (button shouldn't even be visible, but be safe).
                    if not ack:
                        print("  [RACK] Ignoring confirm — rack still exhausted, "
                              "waiting for 'Rack Replaced' acknowledgement.")
                        dashboard.set_state("awaiting_confirmation")
                        continue
                    tip_column_next = 1
                    rack_exhausted = False
                    dashboard.set_tip_rack_status(next_column=1, exhausted=False)
                    print("  [RACK] User acknowledged rack swap — reset to column A1")
                print("  [CONFIRMED] User confirmed — proceeding to next experiment")
                break
            if s in ("stopped", "stop_now"):
                stop_reason = "Campaign stopped by user"
                break
        if stop_reason:
            break

    # ── Campaign finished ─────────────────────────────────────────────────
    dashboard.set_current_params({})   # clear "running" panel when campaign ends
    dashboard.update(optimizer, stop_reason=stop_reason, iter_results=iter_records)

    best_params, best_score = optimizer.get_best()
    print(f"\n{'='*70}")
    print(f"CAMPAIGN COMPLETE — {optimizer.iteration} experiment(s)")
    if stop_reason:
        print(f"  Stop reason  : {stop_reason}")
    print(f"  Best score   : {best_score:.2f}")
    print(f"  Best parameters:")
    for k, v in best_params.items():
        print(f"    {k} = {v}")
    print(f"\n  Protocols  → {cfg.PROTOCOLS_DIR}")
    print(f"  Results    → {cfg.RESULTS_DIR}")
    print(f"  Dashboard  → http://localhost:{cfg.DASHBOARD_PORT}/  (still live)")
    print(f"{'='*70}")

    # Save convergence plot to file (no pop-up — Agg backend is set at import time)
    dashboard.set_robot_status("Campaign finished", "idle")
    try:
        optimizer.plot_convergence(str(cfg.RESULTS_DIR / "convergence_plot.png"))
        print(f"  Plot saved → {cfg.RESULTS_DIR / 'convergence_plot.png'}")
    except Exception as e:
        print(f"  [WARN] Could not save convergence plot: {e}")

    # Keep dashboard alive so user can inspect results
    print("\nPress Ctrl-C to exit (dashboard stays open until then).")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop()
        if cam_proc is not None:
            print("[CAMERA] Stopping camera server…")
            cam_proc.terminate()
            try:
                cam_proc.wait(timeout=5)
            except Exception:
                cam_proc.kill()

    return best_params, best_score


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    skip_wizard = "--no-wizard" in sys.argv
    dry_forced  = "--dry-run"   in sys.argv

    if skip_wizard:
        run_campaign(dry_run=dry_forced)
    else:
        # Known robot IPs to probe during connection check
        robot_ips = ["169.254.83.111", "169.254.84.3", "172.26.4.16", "172.26.4.17"]
        if cfg.ROBOT_IP:
            robot_ips = [cfg.ROBOT_IP] + [ip for ip in robot_ips if ip != cfg.ROBOT_IP]

        wizard = SetupWizard(
            port=cfg.DASHBOARD_PORT,
            robot_ips=robot_ips,
            camera_ip=cfg.CAMERA_SERVER_IP,
            camera_port=cfg.CAMERA_SERVER_PORT,
            camera_index=getattr(cfg, "CAMERA_INDEX", 0),
            factors_meta=build_factors_meta(cfg.FACTORS),
            protocols_dir=str(cfg.PROTOCOLS_DIR),
            dry_run_forced=dry_forced,
        )
        wizard.start()
        wc = wizard.wait_for_start()
        wizard.stop()
        print(f"\n  Wizard config received: {wc}\n")
        run_campaign(wizard_config=wc)
