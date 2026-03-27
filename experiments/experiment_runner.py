"""
AutoLab Experiment Runner
==========================
Closed-loop optimization pipeline:

  Bayesian Optimizer
       ↓  suggested parameters
  Protocol Generator  →  saves iter_NNN_*.py to experiments/protocols/
       ↓  protocol file
  OT-2 Controller     →  runs on robot
       ↓  images saved to captured_images/
  Liquid Analyzer     →  saves CSV to experiments/results/iter_NNN/
       ↓  accuracy score
  Bayesian Optimizer  ←  tell(params, accuracy)

Repeats until stopping criteria met or user presses Stop in dashboard.

Usage:
    python experiment_runner.py              # real robot run
    python experiment_runner.py --dry-run    # no robot, random accuracy (for testing)
"""

import csv as _csv
import shutil
import sys
import time
from pathlib import Path


# ── Results CSV ───────────────────────────────────────────────────────────────

def _write_results_csv(csv_path: Path, iteration: int, phase: str,
                        params: dict, accuracy: float, ei_val,
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
                + ["accuracy_%", "ei", "best_so_far_%", "protocol_file"]
            )
        w.writerow(
            [iteration, time.strftime("%Y-%m-%d %H:%M:%S"), phase]
            + [params[n] for n in factor_names]
            + [
                f"{accuracy:.2f}",
                f"{ei_val:.6f}" if ei_val is not None else "",
                f"{best_so_far:.2f}",
                protocol_file,
            ]
        )

# Set non-interactive matplotlib backend BEFORE any import that loads matplotlib.pyplot
# This prevents plot_convergence() from opening a pop-up window
import matplotlib
matplotlib.use('Agg')

# ── Path setup ────────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = Path(__file__).parent
ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
# Add simulation/ directly so simulation.py can resolve `from dashboard import ...`
sys.path.insert(0, str(ROOT / "simulation"))
# Add root for OT2_operation and camera imports
sys.path.insert(0, str(ROOT))

import campaign_config as cfg
from campaign_dashboard import CampaignDashboard
from protocol_generator import generate_protocol
from setup_wizard import SetupWizard, build_factors_meta
from simulation import DOEOptimizer, simulate_qc_checks_pre_experiment, simulate_qc_checks_per_iteration


# ── Accuracy computation ──────────────────────────────────────────────────────

def compute_accuracy(csv_path: Path) -> float:
    """
    Compute accuracy from liquid_analysis.csv.

    Accuracy = % of images where detection matches expectation:
      before_aspirate → NO   (empty tips)
      after_aspirate  → YES  (liquid loaded)
      before_dispense → YES  (liquid still in tips)
      after_dispense  → NO   (clean dispense — the key quality metric)
    """
    expected = {
        "before_aspirate": "NO",
        "after_aspirate":  "YES",
        "before_dispense": "YES",
        "after_dispense":  "NO",
    }
    correct = total = 0
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                action = row.get("action", "")
                if action in expected:
                    total += 1
                    if row.get("liquid_detected") == expected[action]:
                        correct += 1
    except Exception as e:
        print(f"    [WARN] Could not read analysis CSV: {e}")
        return 0.0
    acc = (correct / total * 100) if total > 0 else 0.0
    print(f"    Analysis: {correct}/{total} images correct → {acc:.1f}%")
    return acc


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


# ── Liquid analysis ───────────────────────────────────────────────────────────

def run_analysis(image_dir: Path, output_dir: Path) -> float:
    """Run liquid analysis on `image_dir` and return accuracy score 0–100."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from camera.analysis_script.liquid_analysis import analyze
        analyze(str(image_dir), str(output_dir), cfg.WEIGHTS_PATH, conf=0.35, threshold=25)
        csv_path = output_dir / "liquid_analysis.csv"
        if csv_path.exists():
            return compute_accuracy(csv_path)
        print("    [WARN] No analysis CSV produced.")
        return 0.0
    except ImportError as e:
        print(f"    [WARN] Could not import liquid_analysis ({e}).")
        return 0.0
    except Exception as e:
        print(f"    [WARN] Analysis failed: {e}")
        return 0.0


# ── Interruptible OT-2 run ────────────────────────────────────────────────────

def run_ot2_interruptible(robot, protocol_path: Path, run_time_params: dict,
                           dashboard, iteration: int) -> str:
    """
    Upload and start a protocol, polling status every 5 s.
    Checks dashboard state on every poll:
      - "stop_now" → immediately stops the robot and returns "stopped_by_user"
    Returns the final OT-2 run status string.
    """
    terminal = {"succeeded", "failed", "stopped"}

    print(f"  Uploading protocol…")
    dashboard.set_robot_status("Uploading protocol", "uploading", "", iteration)
    protocol_id = robot.upload_protocol(str(protocol_path))

    print(f"  Creating run…")
    run_id = robot.create_run(protocol_id, run_time_params)
    robot.start_run(run_id)
    dashboard.set_robot_status(f"OT-2 running — Iter {iteration:03d}", "running", run_id, iteration)

    last_status = None
    while True:
        # ── Check for immediate stop ──────────────────────────────────
        if dashboard.get_state() == "stop_now":
            print("\n  [STOP NOW] Halting robot immediately…")
            robot.stop_run(run_id)
            dashboard.set_robot_status("Stopped by user", "stopped", run_id, iteration)
            return "stopped_by_user"

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
        dry_run:       Skip OT-2 and use random accuracy scores.
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
    camera_ip    = wc.get("camera_ip")   or cfg.CAMERA_SERVER_IP
    camera_port  = wc.get("camera_port") or cfg.CAMERA_SERVER_PORT
    robot_ip     = wc.get("robot_ip")    or cfg.ROBOT_IP

    print("=" * 70)
    print("AutoLab Experiment Campaign")
    print(f"  Mode     : {'DRY RUN (no robot, random scores)' if dry_run else 'LIVE'}")
    print(f"  Factors  : {[f.name for f in factors]}")
    if fixed_params:
        print(f"  Fixed    : {fixed_params}")
    print(f"  Initial  : {n_initial} {'custom' if initial_mode == 'custom' else 'LHS'} points")
    print(f"  Max      : {max_iterations} experiments")
    print(f"  Conv.    : <{convergence_tol}% improvement over {convergence_window} BO iters")
    print(f"  Dashboard: http://localhost:{cfg.DASHBOARD_PORT}/")
    print("=" * 70)

    # ── Directories ──────────────────────────────────────────────────────
    cfg.PROTOCOLS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = DOEOptimizer(
        factors=factors,
        n_initial=n_initial,
        max_iterations=max_iterations,
        xi=cfg.XI,
        convergence_tol=convergence_tol,
        convergence_window=convergence_window,
        csv_path=str(cfg.RESULTS_DIR / "campaign_log.csv"),
        random_seed=cfg.RANDOM_SEED,
        dashboard_path=str(cfg.RESULTS_DIR / "_sim_dashboard.html"),
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

    # ── Robot ─────────────────────────────────────────────────────────────
    robot = None
    if not dry_run:
        from OT2_operation.ot2_controller import OT2
        try:
            robot = OT2(ip=robot_ip)
            robot.lights(lights_on)
        except ConnectionError as e:
            print(f"\n[ERROR] {e}")
            print("  Tip: run with --dry-run to test without a robot.")
            dashboard.stop()
            sys.exit(1)

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

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:

        # ── Control state check ───────────────────────────────────────
        state = dashboard.get_state()

        if state in ("stopped", "stop_now"):
            stop_reason = "Campaign stopped by user"
            print(f"\n[STOP] {stop_reason}")
            break

        if state == "paused":
            print("  [PAUSED] Waiting for Resume in dashboard…")
            while True:
                time.sleep(3)
                s = dashboard.get_state()
                if s == "stopped":
                    stop_reason = "Campaign stopped by user"
                    break
                if s == "running":
                    print("  [RESUMED]")
                    break
            if stop_reason:
                break
            continue

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
        n     = optimizer.iteration
        phase = "INITIAL" if n < n_initial else "BO"

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

        print(f"\n{'='*60}")
        print(f"Iteration {n:03d} | Phase: {phase}")
        for k, v in params.items():
            print(f"  {k} = {v}")
        if ei_val is not None:
            print(f"  EI = {ei_val:.4f}")

        # Show upcoming params in dashboard while experiment runs
        dashboard.set_robot_status(f"Planning Iter {n:03d} — generating protocol", "idle", "", n)
        dashboard.update(optimizer, next_suggestion=params, iter_results=iter_records)

        # ── Generate protocol file ────────────────────────────────────
        print(f"\n  Generating protocol file…")
        protocol_path = generate_protocol(
            params, n,
            protocols_dir=cfg.PROTOCOLS_DIR,
            num_columns=cfg.NUM_COLUMNS,
            transfer_volume=cfg.TRANSFER_VOLUME,
            camera_ip=camera_ip,
            camera_port=camera_port,
            settle_seconds=cfg.SETTLE_SECONDS,
            camera_height_mm=cfg.CAMERA_HEIGHT_MM,
        )

        # ── Run experiment ────────────────────────────────────────────
        iter_dir = cfg.RESULTS_DIR / f"iter_{n:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        accuracy = 0.0

        # Show parameters being executed now
        dashboard.set_current_params(params)

        if dry_run:
            dashboard.set_robot_status(f"DRY RUN — Iter {n:03d} simulating", "running", "", n)
            print("  [DRY RUN] Skipping OT-2 — generating random accuracy")
            time.sleep(0.5)
            accuracy = round(random.uniform(40, 99), 2)
            print(f"  Simulated accuracy: {accuracy:.1f}%")
            dashboard.set_robot_status(f"DRY RUN — Iter {n:03d} complete ({accuracy:.1f}%)", "succeeded", "", n)

        else:
            run_time_params = {
                "camera_server_ip":   cfg.CAMERA_SERVER_IP,
                "camera_server_port": cfg.CAMERA_SERVER_PORT,
                "capture_enabled":    True,
                "settle_seconds":     cfg.SETTLE_SECONDS,
                "camera_height_mm":   cfg.CAMERA_HEIGHT_MM,
            }

            print(f"\n  Running on OT-2…")
            run_start = time.time()
            ot2_result = "failed"
            try:
                ot2_result = run_ot2_interruptible(
                    robot, protocol_path, run_time_params, dashboard, n
                )
            except Exception as e:
                print(f"  [ERROR] OT-2 run failed: {e}")
                dashboard.set_robot_status(f"OT-2 error — Iter {n:03d}", "failed", "", n)

            if ot2_result == "stopped_by_user":
                stop_reason = "Campaign stopped by user (Stop Now)"
                # Still record the iteration with accuracy=0
                optimizer.per_iteration_qc.append(simulate_qc_checks_per_iteration(0))
                optimizer.tell(params, 0.0, ei_val, phase)
                record = {"protocol_file": protocol_path.name}
                iter_records.append(record)
                _write_results_csv(
                    csv_path=cfg.RESULTS_DIR / "results.csv",
                    iteration=n, phase=phase, params=params,
                    accuracy=0.0, ei_val=ei_val,
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

                # ── Liquid analysis ───────────────────────────────────
                dashboard.set_robot_status(f"Analysing images — Iter {n:03d}", "succeeded", "", n)
                print("\n  Running liquid analysis…")
                analysis_dir = iter_dir / "analysis"
                accuracy = run_analysis(image_dir, analysis_dir)
            else:
                accuracy = 0.0

        # ── Update optimizer ──────────────────────────────────────────
        optimizer.per_iteration_qc.append(
            simulate_qc_checks_per_iteration(accuracy)
        )
        optimizer.tell(params, accuracy, ei_val, phase)

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
            phase=phase,
            params=params,
            accuracy=accuracy,
            ei_val=ei_val,
            best_so_far=best_so_far,
            protocol_file=protocol_path.name,
        )

        # ── Dashboard update ──────────────────────────────────────────
        dashboard.set_robot_status(f"Iter {n:03d} complete — accuracy {accuracy:.1f}%", "idle", "", n)
        dashboard.update(optimizer, iter_results=iter_records)
        print(f"\n  Best so far: {best_so_far:.1f}%")

        # ── Post-tell convergence check (catches convergence immediately) ─
        if phase == "BO":
            converged, conv_reason = optimizer.check_convergence()
            if converged:
                stop_reason = conv_reason
                print(f"\n*** CONVERGED: {stop_reason} ***")
                break

    # ── Campaign finished ─────────────────────────────────────────────────
    dashboard.set_current_params({})   # clear "running" panel when campaign ends
    dashboard.update(optimizer, stop_reason=stop_reason, iter_results=iter_records)

    best_params, best_acc = optimizer.get_best()
    print(f"\n{'='*70}")
    print(f"CAMPAIGN COMPLETE — {optimizer.iteration} experiment(s)")
    if stop_reason:
        print(f"  Stop reason  : {stop_reason}")
    print(f"  Best accuracy: {best_acc:.2f}%")
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

    return best_params, best_acc


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    skip_wizard = "--no-wizard" in sys.argv
    dry_forced  = "--dry-run"   in sys.argv

    if skip_wizard:
        run_campaign(dry_run=dry_forced)
    else:
        # Known robot IPs to probe during connection check
        robot_ips = ["169.254.84.3", "172.26.4.16", "172.26.4.17"]
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
