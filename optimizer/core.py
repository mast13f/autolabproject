"""
Pipetting DOE Optimizer
=======================
Bayesian Optimization with Gaussian Process surrogate for liquid-handling
parameter tuning.  Supports liquid-type tracking, persistent cross-run
training logs, and an integrated image-analysis path via measure_and_tell().

Factor ranges are calibrated for the OT-2 p20 multi-channel pipette:
  aspirate_speed  0–31 µL/s
  dispense_speed  1–31 µL/s
  air_gap         categorical {0, 1}
  blow_out        categorical {0, 1}
"""

from __future__ import annotations

import csv
import itertools
import os
import random
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Quality-Control Metrics (unchanged public interface)
# ---------------------------------------------------------------------------

@dataclass
class QCCheck:
    name: str
    passed: bool
    value: str
    rationale: str
    timing: str  # when the check is performed


def simulate_qc_checks_pre_experiment() -> List[QCCheck]:
    """Simulate pre-experiment QC checks.  All pass within acceptable ranges."""
    checks = []
    days_since_cal = random.randint(0, 18)
    checks.append(QCCheck(
        name="Opentron Calibrated",
        passed=True,
        value=f"Calibrated {days_since_cal} days ago",
        rationale="If the Opentrons is not calibrated properly, it could cause problems "
                  "like not picking up tips or not aspirating/dispensing at the right height.",
        timing="Before experiment begins",
    ))
    checks.append(QCCheck(
        name="Pipette Tips Condition",
        passed=True,
        value="8/8 tips in good condition",
        rationale="If the pipette tips are in bad condition they may not properly aspirate "
                  "or dispense liquid, leading to skewed results.",
        timing="Before experiment begins",
    ))
    dye_concentration = round(random.uniform(0.5, 2.0), 2)
    checks.append(QCCheck(
        name="Liquid Is Dyed",
        passed=True,
        value=f"Dye concentration: {dye_concentration} mg/mL",
        rationale="If the liquid is not dyed, score measurement may be off since it "
                  "would be difficult to detect remaining liquid in the tips.",
        timing="Before experiment begins",
    ))
    return checks


def simulate_qc_checks_per_iteration(score: float) -> List[QCCheck]:
    """Simulate per-iteration QC checks for a single experiment."""
    checks = []
    checks.append(QCCheck(
        name="Liquid in Tips Before Pipetting",
        passed=True,
        value="No liquid detected",
        rationale="Ground truth of what the pipette looks like with no liquid in it.",
        timing="Before aspirating liquid",
    ))
    if score > 70:
        residual = random.choice(["No liquid detected", "Trace amount detected"])
    elif score > 40:
        residual = random.choice(["Trace amount detected", "Small amount detected"])
    else:
        residual = random.choice(["Small amount detected", "Moderate amount detected"])
    checks.append(QCCheck(
        name="Liquid in Tips After Dispensing",
        passed=True,
        value=residual,
        rationale="Liquid presence in tip after dispensing measures how well the parameters "
                  "work. Little to no liquid means optimal parameters.",
        timing="After dispensing liquid",
    ))
    return checks


# ---------------------------------------------------------------------------
# Factor definition
# ---------------------------------------------------------------------------

@dataclass
class Factor:
    name: str
    kind: str  # "continuous" or "categorical"
    low: float
    high: float
    levels: Optional[List[float]] = None  # for categorical factors


PIPETTING_FACTORS = [
    Factor("aspirate_speed", "continuous",  0.0, 31.0),
    Factor("dispense_speed", "continuous",  1.0, 31.0),
    Factor("air_gap",        "categorical", 0.0,  1.0, levels=[0.0, 1.0]),
    Factor("blow_out",       "categorical", 0.0,  1.0, levels=[0.0, 1.0]),
]

# Backward-compatible alias
FACTORS = PIPETTING_FACTORS


# ---------------------------------------------------------------------------
# Liquid registry
# ---------------------------------------------------------------------------

KNOWN_LIQUIDS = ["water", "glycerol", "ethanol"]

LIQUID_CONCENTRATION_RULES: Dict[str, Tuple[float, float]] = {
    "water":    (100.0, 100.0),
    "glycerol": (0.0,   100.0),
    "ethanol":  (0.0,   100.0),
}


def validate_liquid(liquid_type: str, concentration_pct: float) -> Tuple[str, float]:
    """Validate liquid type and concentration against known rules."""
    liquid_type = liquid_type.lower().strip()
    if liquid_type not in KNOWN_LIQUIDS:
        raise ValueError(f"Unknown liquid '{liquid_type}'. Known: {KNOWN_LIQUIDS}")
    lo, hi = LIQUID_CONCENTRATION_RULES[liquid_type]
    if not (lo <= concentration_pct <= hi):
        raise ValueError(
            f"Concentration {concentration_pct}% out of range [{lo}, {hi}] "
            f"for {liquid_type}."
        )
    return liquid_type, concentration_pct


# ---------------------------------------------------------------------------
# Training-log CSV
# ---------------------------------------------------------------------------

TRAINING_CSV = "training_data.csv"

TRAINING_FIELDNAMES = [
    "run_id", "timestamp_utc", "liquid_type", "concentration_pct",
    "iteration", "phase",
    "aspirate_speed", "dispense_speed", "air_gap", "blow_out",
    "score", "ei_value", "best_so_far",
    "px_after_dispense",
]


def _init_training_csv(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=TRAINING_FIELDNAMES).writeheader()


def _append_training_row(path: str, row: dict):
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=TRAINING_FIELDNAMES, extrasaction="ignore").writerow(row)


# ---------------------------------------------------------------------------
# Initial design — Latin Hypercube Sampling
# ---------------------------------------------------------------------------

def generate_initial_design(
    factors: List[Factor], n_points: int = 5, seed: int = 42,
) -> np.ndarray:
    """Generate an initial experimental design via Latin Hypercube Sampling.

    Returns an ``(n_points × n_factors)`` array with values in raw parameter space.
    """
    n_factors = len(factors)
    sampler = LatinHypercube(d=n_factors, seed=seed)
    unit_samples = sampler.random(n=n_points)
    design = np.empty_like(unit_samples)
    for j, f in enumerate(factors):
        scaled = f.low + unit_samples[:, j] * (f.high - f.low)
        if f.kind == "categorical" and f.levels is not None:
            levels = np.array(f.levels)
            for i in range(n_points):
                design[i, j] = levels[np.argmin(np.abs(levels - scaled[i]))]
        else:
            design[:, j] = scaled
    return design


# ---------------------------------------------------------------------------
# Gaussian Process surrogate
# ---------------------------------------------------------------------------

def build_gp(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    """Fit a GP with Matérn 5/2 kernel to observed (normalized) data."""
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X, y)
    return gp


# ---------------------------------------------------------------------------
# Acquisition function — Expected Improvement
# ---------------------------------------------------------------------------

def expected_improvement(
    X_candidates: np.ndarray,
    gp: GaussianProcessRegressor,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Compute Expected Improvement at each candidate point."""
    mu, sigma = gp.predict(X_candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    Z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei


# ---------------------------------------------------------------------------
# Next-point suggestion (mixed continuous + categorical space)
# ---------------------------------------------------------------------------

def suggest_next_point(
    gp: GaussianProcessRegressor,
    factors: List[Factor],
    y_best: float,
    xi: float = 0.01,
    n_random: int = 5000,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, float]:
    """Find the point maximising Expected Improvement.

    Enumerates all categorical combinations, then optimises the continuous
    dimensions via L-BFGS-B from the best random seeds.
    """
    if rng is None:
        rng = np.random.default_rng()
    cont_idx = [i for i, f in enumerate(factors) if f.kind == "continuous"]
    cat_idx = [i for i, f in enumerate(factors) if f.kind == "categorical"]
    cat_combos = list(itertools.product(*[factors[i].levels for i in cat_idx]))
    n = len(factors)
    best_pt, best_ei = None, -np.inf

    for combo in cat_combos:
        cands = np.empty((n_random, n))
        for j in cont_idx:
            cands[:, j] = rng.uniform(0.0, 1.0, size=n_random)
        for k, ci in enumerate(cat_idx):
            cands[:, ci] = (combo[k] - factors[ci].low) / (factors[ci].high - factors[ci].low)
        ei_vals = expected_improvement(cands, gp, y_best, xi)
        for idx in np.argsort(ei_vals)[-5:]:
            x0 = cands[idx].copy()
            bounds = [
                (0.0, 1.0) if j in cont_idx else (x0[j], x0[j])
                for j in range(n)
            ]

            def neg_ei(x):
                return -expected_improvement(x.reshape(1, -1), gp, y_best, xi)[0]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(neg_ei, x0, bounds=bounds, method="L-BFGS-B")
            if -res.fun > best_ei:
                best_ei = -res.fun
                best_pt = res.x.copy()

    return best_pt, best_ei


# ---------------------------------------------------------------------------
# PipettingDOEOptimizer
# ---------------------------------------------------------------------------

class PipettingDOEOptimizer:
    """Bayesian optimiser for OT-2 pipetting parameters.

    Attributes consumed by the runner / dashboard:
        X_raw, y_observed, ei_values, phases, iteration,
        factors, n_initial, max_iterations, convergence_tol,
        convergence_window, pre_experiment_qc, per_iteration_qc
    """

    def __init__(
        self,
        liquid_type: str = "water",
        concentration_pct: float = 100.0,
        factors: List[Factor] | None = None,
        n_initial: int = 5,
        max_iterations: int = 20,
        xi: float = 0.01,
        convergence_tol: float = 1.0,
        convergence_window: int = 5,
        image_threshold: int = 12,
        run_id: str | None = None,
        training_csv: str = TRAINING_CSV,
        random_seed: int = 42,
        # Legacy kwargs accepted but unused (keeps runner construction simple)
        csv_path: str | None = None,
        dashboard_path: str | None = None,
    ):
        self.liquid_type, self.concentration_pct = validate_liquid(
            liquid_type, concentration_pct,
        )
        self.factors = factors or PIPETTING_FACTORS
        self.n_initial = n_initial
        self.max_iterations = max_iterations
        self.xi = xi
        self.convergence_tol = convergence_tol
        self.convergence_window = convergence_window
        self.image_threshold = image_threshold
        self.training_csv = training_csv
        self.rng = np.random.default_rng(random_seed)

        self.run_id = run_id or (
            f"{liquid_type}_{concentration_pct:.0f}pct_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )

        # ── Observation state ────────────────────────────────────────────
        self.X_raw: List[np.ndarray] = []
        self.y_observed: List[float] = []
        self.ei_values: List[Optional[float]] = []
        self.phases: List[str] = []
        self.gp: Optional[GaussianProcessRegressor] = None
        self.iteration: int = 0
        self.done: bool = False
        self._stop_reason: Optional[str] = None

        # ── QC metrics (consumed by dashboard) ───────────────────────────
        self.pre_experiment_qc: List[QCCheck] = []
        self.per_iteration_qc: List[List[QCCheck]] = []

        # ── Initial design ───────────────────────────────────────────────
        self._initial_design = generate_initial_design(
            self.factors, n_initial, seed=random_seed,
        )

        # ── Training CSV ─────────────────────────────────────────────────
        _init_training_csv(training_csv)

        # Legacy CSV (runner may pass a path for its own campaign log)
        self._csv_path = csv_path
        if csv_path:
            self._init_legacy_csv(csv_path)

    # ── Legacy CSV (backward compat) ─────────────────────────────────────

    def _init_legacy_csv(self, path: str):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = (
                ["iteration", "phase"]
                + [fac.name for fac in self.factors]
                + ["score", "ei_value", "best_so_far"]
            )
            writer.writerow(header)

    # ── Normalisation helpers ────────────────────────────────────────────

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        out = np.empty_like(X)
        for j, f in enumerate(self.factors):
            w = f.high - f.low
            out[:, j] = 0.0 if w == 0 else (X[:, j] - f.low) / w
        return out

    def _denormalize(self, X: np.ndarray) -> np.ndarray:
        out = np.empty_like(X)
        for j, f in enumerate(self.factors):
            out[:, j] = f.low + X[:, j] * (f.high - f.low)
        return out

    # ── Dict conversions ─────────────────────────────────────────────────

    def _to_dict(self, x_raw: np.ndarray) -> dict:
        return {
            f.name: (
                int(round(x_raw[j]))
                if f.kind == "categorical"
                else round(float(x_raw[j]), 2)
            )
            for j, f in enumerate(self.factors)
        }

    # Dashboard calls _raw_to_dict
    _raw_to_dict = _to_dict

    # ── Logging ──────────────────────────────────────────────────────────

    def _log(self, params: dict, score: float, ei: Optional[float],
             phase: str, px: dict):
        best = max(self.y_observed)
        ei_s = f"  EI={ei:.4f}" if ei is not None else ""
        print(
            f"[Iter {self.iteration:03d} | {phase:8s}] "
            f"{' '.join(f'{k}={v}' for k, v in params.items())}  "
            f"=> score={score:.2f}  best_so_far={best:.2f}{ei_s}"
        )
        _append_training_row(self.training_csv, {
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "liquid_type": self.liquid_type,
            "concentration_pct": self.concentration_pct,
            "iteration": self.iteration,
            "phase": phase,
            **params,
            "score": round(score, 4),
            "ei_value": round(ei, 6) if ei is not None else "",
            "best_so_far": round(best, 4),
            "px_after_dispense": px.get("after_dispense", ""),
        })
        # Legacy CSV
        if self._csv_path:
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [self.iteration, phase]
                    + [params[fac.name] for fac in self.factors]
                    + [
                        f"{score:.4f}",
                        f"{ei:.6f}" if ei is not None else "",
                        f"{best:.4f}",
                    ]
                )

    # ── suggest ──────────────────────────────────────────────────────────

    def suggest(self) -> Tuple[dict, Optional[float]]:
        """Return next ``(params_dict, ei_value)``."""
        if self.iteration < self.n_initial:
            return self._to_dict(self._initial_design[self.iteration]), None

        X_norm = self._normalize(np.array(self.X_raw))
        y = np.array(self.y_observed)
        self.gp = build_gp(X_norm, y)

        next_norm, ei_val = suggest_next_point(
            self.gp, self.factors, y.max(), self.xi, rng=self.rng,
        )
        next_raw = self._denormalize(next_norm.reshape(1, -1))[0]

        for j, f in enumerate(self.factors):
            if f.kind == "categorical" and f.levels:
                lvls = np.array(f.levels)
                next_raw[j] = lvls[np.argmin(np.abs(lvls - next_raw[j]))]

        return self._to_dict(next_raw), ei_val

    # ── tell ─────────────────────────────────────────────────────────────

    def tell(self, params: dict, score: float, ei: Optional[float],
             px: dict | None = None):
        """Record an observation.  Phase is computed automatically."""
        phase = "INITIAL" if self.iteration < self.n_initial else "BO"
        self.X_raw.append(
            np.array([params[f.name] for f in self.factors], dtype=float),
        )
        self.y_observed.append(score)
        self.ei_values.append(ei)
        self.phases.append(phase)
        self._log(params, score, ei, phase, px or {})
        self.iteration += 1
        self._check_done()

    # ── measure_and_tell (integrated analysis path) ──────────────────────

    def measure_and_tell(
        self,
        image_dir: str,
        output_dir: str,
        params: dict,
        ei: Optional[float],
    ) -> Optional[float]:
        """Run image analysis then record the result.

        Returns the computed score, or ``None`` on failure.
        """
        from autolab.image_analysis import measure_from_images, aggregate_score

        print(f"\n  [Measuring] {image_dir}")
        col_results = measure_from_images(
            image_dir, output_dir, threshold=self.image_threshold,
        )
        score = aggregate_score(col_results)
        if score is None:
            print("  [ERROR] Could not compute score — check image filenames.")
            return None

        px: dict = {}
        vals = [
            v["px_after_dispense"]
            for v in col_results.values()
            if v.get("px_after_dispense") is not None
        ]
        if vals:
            px["after_dispense"] = int(np.mean(vals))

        self.tell(params, score, ei, px=px)
        return score

    # ── Convergence check ────────────────────────────────────────────────

    def _check_done(self):
        if self.iteration >= self.max_iterations:
            self._stop_reason = (
                f"Experiment cap: {self.max_iterations} total experiments."
            )
            self.done = True
            return
        bo_idx = [i for i, p in enumerate(self.phases) if p == "BO"]
        if len(bo_idx) >= self.convergence_window:
            older = bo_idx[:-self.convergence_window]
            if older:
                improvement = max(self.y_observed) - max(
                    self.y_observed[i] for i in older
                )
                if improvement < self.convergence_tol:
                    self._stop_reason = (
                        f"Converged: improvement {improvement:.2f} < "
                        f"{self.convergence_tol} over last "
                        f"{self.convergence_window} BO iters."
                    )
                    self.done = True

    def check_convergence(self) -> Tuple[bool, Optional[str]]:
        """Public interface consumed by the runner.

        Returns ``(converged, reason_string)`` compatible with the dashboard.
        """
        bo_indices = [i for i, p in enumerate(self.phases) if p == "BO"]
        if len(bo_indices) < self.convergence_window:
            return False, None
        older = bo_indices[:-self.convergence_window]
        if not older:
            return False, None
        best_before_window = max(self.y_observed[i] for i in older)
        best_overall = max(self.y_observed)
        improvement = best_overall - best_before_window
        if improvement < self.convergence_tol:
            reason = (
                f"Score converged: best improved only {improvement:.2f}% "
                f"(< {self.convergence_tol}% threshold) over last "
                f"{self.convergence_window} BO iterations"
            )
            return True, reason
        return False, None

    # ── Best result ──────────────────────────────────────────────────────

    def get_best(self) -> Tuple[dict, float]:
        """Return ``(best_params_dict, best_score)``."""
        idx = int(np.argmax(self.y_observed))
        return self._to_dict(self.X_raw[idx]), self.y_observed[idx]

    # ── Convergence plot ─────────────────────────────────────────────────

    def plot_convergence(self, save_path: str = "doe_convergence.png"):
        """Save a two-panel convergence plot to *save_path*."""
        iters = np.arange(len(self.y_observed))
        scores = np.array(self.y_observed)
        best = np.maximum.accumulate(scores)
        colors = [
            "#1f77b4" if p == "INITIAL" else "#ff7f0e" for p in self.phases
        ]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        ax1.scatter(
            iters, scores, c=colors, s=50, zorder=3,
            edgecolors="k", linewidths=0.5,
        )
        ax1.axvline(
            x=self.n_initial - 0.5, color="gray", linestyle="--", alpha=0.7,
        )
        ax1.set_ylabel("Score")
        ax1.set_title(
            f"{self.liquid_type.title()} {self.concentration_pct:.0f}% "
            f"— DOE Convergence"
        )
        from matplotlib.patches import Patch
        ax1.legend(
            handles=[
                Patch(facecolor="#1f77b4", edgecolor="k", label="Initial (LHS)"),
                Patch(facecolor="#ff7f0e", edgecolor="k", label="Bayesian Opt"),
            ],
            loc="lower right",
        )
        ax1.grid(True, alpha=0.3)

        ax2.plot(iters, best, "r-o", markersize=4, linewidth=2)
        ax2.axvline(
            x=self.n_initial - 0.5, color="gray", linestyle="--", alpha=0.7,
        )
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Best Score")
        ax2.set_title("Cumulative Best Score")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\nConvergence plot saved to: {save_path}")

    # ── Summary ──────────────────────────────────────────────────────────

    def summary(self):
        best_params, best_score = self.get_best()
        print(f"\n{'=' * 65}")
        print(f"  DONE  — {self.liquid_type} @ {self.concentration_pct:.1f}%")
        print(f"  Iterations  : {self.iteration}")
        print(f"  Stop reason : {self._stop_reason}")
        print(f"  Best score  : {best_score:.2f}/100")
        print(f"  Best params : {best_params}")
        print(f"  Training log: {self.training_csv}")
        print(f"{'=' * 65}\n")


# Backward-compatible alias
DOEOptimizer = PipettingDOEOptimizer
