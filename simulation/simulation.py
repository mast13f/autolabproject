import os
import random
import csv
import itertools
import webbrowser
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable

import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import matplotlib.pyplot as plt

from dashboard import CampaignDashboard


# ---------------------------------------------------------------------------
# Quality Control Metrics
# ---------------------------------------------------------------------------

@dataclass
class QCCheck:
    name: str
    passed: bool
    value: str
    rationale: str
    timing: str  # when the check is performed


def simulate_qc_checks_pre_experiment() -> List[QCCheck]:
    """Simulate pre-experiment QC checks. All pass within acceptable ranges."""
    checks = []

    # 1. Opentron calibrated (must be within last 3 weeks)
    days_since_cal = random.randint(0, 18)  # always < 21 days
    checks.append(QCCheck(
        name="Opentron Calibrated",
        passed=True,
        value=f"Calibrated {days_since_cal} days ago",
        rationale="If the Opentrons is not calibrated properly, it could cause problems "
                  "like not picking up tips or not aspirating/dispensing at the right height.",
        timing="Before experiment begins",
    ))

    # 2. Pipette tips in good condition
    n_tips = random.randint(8, 8)  # all 8 tips good
    checks.append(QCCheck(
        name="Pipette Tips Condition",
        passed=True,
        value=f"{n_tips}/8 tips in good condition",
        rationale="If the pipette tips are in bad condition they may not properly aspirate "
                  "or dispense liquid, leading to skewed results.",
        timing="Before experiment begins",
    ))

    # 3. Liquid is dyed
    dye_concentration = round(random.uniform(0.5, 2.0), 2)
    checks.append(QCCheck(
        name="Liquid Is Dyed",
        passed=True,
        value=f"Dye concentration: {dye_concentration} mg/mL",
        rationale="If the liquid is not dyed, accuracy measurement may be off since it "
                  "would be difficult to detect remaining liquid in the tips.",
        timing="Before experiment begins",
    ))

    return checks


def simulate_qc_checks_per_iteration(accuracy: float) -> List[QCCheck]:
    """Simulate per-iteration QC checks for a single experiment."""
    checks = []

    # 4. Liquid presence in tips before pipetting (ground truth — should be NO liquid)
    checks.append(QCCheck(
        name="Liquid in Tips Before Pipetting",
        passed=True,
        value="No liquid detected",
        rationale="Ground truth of what the pipette looks like with no liquid in it.",
        timing="Before aspirating liquid",
    ))

    # 5. Liquid presence in tips after dispensing
    # Higher accuracy => less residual liquid (better dispense)
    if accuracy > 70:
        residual = random.choice(["No liquid detected", "Trace amount detected"])
    elif accuracy > 40:
        residual = random.choice(["Trace amount detected", "Small amount detected"])
    else:
        residual = random.choice(["Small amount detected", "Moderate amount detected"])

    checks.append(QCCheck(
        name="Liquid in Tips After Dispensing",
        passed=True,  # always "passes" as a data point (not a gate)
        value=residual,
        rationale="Liquid presence in tip after dispensing measures how well the parameters "
                  "work. Little to no liquid means optimal parameters.",
        timing="After dispensing liquid",
    ))

    return checks


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate_accuracy(aspirate_speed, dispense_speed, air_gap, blow_out):
    """Simulate a pipetting accuracy level based on given parameters.

    Args:
        aspirate_speed: Speed of aspiration (0-200).
        dispense_speed: Speed of dispensing (0-200).
        air_gap: Whether to use an air gap (0 or 1).
        blow_out: Whether to use blow out (0 or 1).

    Returns:
        A random accuracy level between 0 and 100.
    """
    random.seed()
    accuracy = random.uniform(0, 100)
    return accuracy


# ---------------------------------------------------------------------------
# Factor definitions
# ---------------------------------------------------------------------------

@dataclass
class Factor:
    name: str
    kind: str  # "continuous" or "categorical"
    low: float
    high: float
    levels: Optional[List[float]] = None  # for categorical factors


FACTORS = [
    Factor("aspirate_speed", "continuous", 0.0, 200.0),
    Factor("dispense_speed", "continuous", 0.0, 200.0),
    Factor("air_gap", "categorical", 0.0, 1.0, levels=[0.0, 1.0]),
    Factor("blow_out", "categorical", 0.0, 1.0, levels=[0.0, 1.0]),
]


# ---------------------------------------------------------------------------
# Initial design — Latin Hypercube Sampling
# ---------------------------------------------------------------------------

def generate_initial_design(factors: List[Factor], n_points: int = 5,
                            seed: int = 42) -> np.ndarray:
    """Generate an initial experimental design via Latin Hypercube Sampling.

    Returns an (n_points x n_factors) array with values in raw parameter space.
    """
    n_factors = len(factors)
    sampler = LatinHypercube(d=n_factors, seed=seed)
    unit_samples = sampler.random(n=n_points)  # shape (n_points, n_factors)

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
    """Fit a GP with Matern 5/2 kernel to observed (normalized) data."""
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=42
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X, y)
    return gp


# ---------------------------------------------------------------------------
# Acquisition function — Expected Improvement
# ---------------------------------------------------------------------------

def expected_improvement(X_candidates: np.ndarray, gp: GaussianProcessRegressor,
                         y_best: float, xi: float = 0.01) -> np.ndarray:
    """Compute Expected Improvement at each candidate point."""
    mu, sigma = gp.predict(X_candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    Z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei


# ---------------------------------------------------------------------------
# Next-point suggestion (mixed-space optimization)
# ---------------------------------------------------------------------------

def suggest_next_point(gp: GaussianProcessRegressor, factors: List[Factor],
                       y_best: float, xi: float = 0.01,
                       n_random: int = 5000, rng: np.random.Generator = None
                       ) -> Tuple[np.ndarray, float]:
    """Find the point maximizing Expected Improvement.

    Strategy: enumerate all categorical combos (2^2 = 4), for each combo
    optimize the continuous dims via L-BFGS-B from the best random seeds.
    """
    if rng is None:
        rng = np.random.default_rng()

    cont_indices = [i for i, f in enumerate(factors) if f.kind == "continuous"]
    cat_indices = [i for i, f in enumerate(factors) if f.kind == "categorical"]
    cat_levels = [factors[i].levels for i in cat_indices]
    cat_combos = list(itertools.product(*cat_levels))

    n_factors = len(factors)
    best_point = None
    best_ei = -np.inf

    for combo in cat_combos:
        # Generate random continuous candidates
        candidates = np.empty((n_random, n_factors))
        for j in cont_indices:
            f = factors[j]
            candidates[:, j] = rng.uniform(
                (f.low - f.low) / (f.high - f.low),
                1.0,
                size=n_random
            )  # normalized [0, 1]
        for k, ci in enumerate(cat_indices):
            f = factors[ci]
            candidates[:, ci] = (combo[k] - f.low) / (f.high - f.low)

        ei_vals = expected_improvement(candidates, gp, y_best, xi)
        top_k = np.argsort(ei_vals)[-5:]

        for idx in top_k:
            x0 = candidates[idx].copy()

            # Bounds: continuous dims in [0,1], categorical dims fixed
            bounds_opt = []
            for j in range(n_factors):
                if j in cont_indices:
                    bounds_opt.append((0.0, 1.0))
                else:
                    val = x0[j]
                    bounds_opt.append((val, val))  # fixed

            def neg_ei(x):
                return -expected_improvement(x.reshape(1, -1), gp, y_best, xi)[0]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(neg_ei, x0, bounds=bounds_opt, method="L-BFGS-B")

            ei_at_opt = -result.fun
            if ei_at_opt > best_ei:
                best_ei = ei_at_opt
                best_point = result.x.copy()

    return best_point, best_ei


# ---------------------------------------------------------------------------
# DOE Optimizer
# ---------------------------------------------------------------------------

class DOEOptimizer:
    def __init__(self, factors: List[Factor], n_initial: int = 5,
                 max_iterations: int = 20, xi: float = 0.01,
                 convergence_tol: float = 1.0, convergence_window: int = 5,
                 csv_path: str = "doe_log.csv", random_seed: int = 42,
                 dashboard_path: str = "campaign_dashboard.html"):
        self.factors = factors
        self.n_initial = n_initial
        self.max_iterations = max_iterations  # Hard cap: stop after this many experiments
        self.xi = xi
        self.convergence_tol = convergence_tol  # Stop if best improves < this % over window
        self.convergence_window = convergence_window
        self.csv_path = csv_path
        self.rng = np.random.default_rng(random_seed)
        self.dashboard = CampaignDashboard(dashboard_path)

        # State
        self.X_raw = []       # list of np.arrays in raw param space
        self.y_observed = []  # accuracy values
        self.ei_values = []   # EI value for each point (None for initial)
        self.phases = []      # "INITIAL" or "BO"
        self.gp = None
        self.iteration = 0

        # QC metrics
        self.pre_experiment_qc = []     # list of QCCheck (run once before experiments)
        self.per_iteration_qc = []      # list of list of QCCheck (one list per iteration)

        # Pre-generate initial design
        self._initial_design = generate_initial_design(
            factors, n_initial, seed=random_seed
        )

        # Initialize CSV
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["iteration", "phase"] + [fac.name for fac in self.factors] + [
                "accuracy", "ei_value", "best_so_far"
            ]
            writer.writerow(header)

    def _normalize(self, X_raw: np.ndarray) -> np.ndarray:
        """Scale raw parameters to [0, 1]."""
        X_norm = np.empty_like(X_raw)
        for j, f in enumerate(self.factors):
            rng_width = f.high - f.low
            if rng_width == 0:
                X_norm[:, j] = 0.0
            else:
                X_norm[:, j] = (X_raw[:, j] - f.low) / rng_width
        return X_norm

    def _denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        """Scale [0, 1] back to raw parameter space."""
        X_raw = np.empty_like(X_norm)
        for j, f in enumerate(self.factors):
            X_raw[:, j] = f.low + X_norm[:, j] * (f.high - f.low)
        return X_raw

    def _raw_to_dict(self, x_raw: np.ndarray) -> dict:
        """Convert a raw parameter array to a named dict."""
        params = {}
        for j, f in enumerate(self.factors):
            val = x_raw[j]
            if f.kind == "categorical":
                val = int(round(val))
            else:
                val = float(round(val, 2))
            params[f.name] = val
        return params

    def _log_iteration(self, params: dict, accuracy: float,
                       ei: Optional[float], phase: str, best_so_far: float):
        """Print to console and append to CSV."""
        # Console
        param_str = "  ".join(f"{k}={v}" for k, v in params.items())
        ei_str = f"  EI={ei:.4f}" if ei is not None else ""
        print(f"[Iter {self.iteration:03d} | {phase:8s}] {param_str}  "
              f"=> accuracy={accuracy:.2f}  best_so_far={best_so_far:.2f}{ei_str}")

        # CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            row = [self.iteration, phase] + [params[fac.name] for fac in self.factors] + [
                f"{accuracy:.4f}",
                f"{ei:.6f}" if ei is not None else "",
                f"{best_so_far:.4f}",
            ]
            writer.writerow(row)

    def suggest(self) -> Tuple[dict, Optional[float]]:
        """Return the next parameter combination and its EI value."""
        if self.iteration < self.n_initial:
            # Initial design phase
            x_raw = self._initial_design[self.iteration]
            return self._raw_to_dict(x_raw), None
        else:
            # Bayesian optimization phase
            X_raw = np.array(self.X_raw)
            X_norm = self._normalize(X_raw)
            y = np.array(self.y_observed)

            self.gp = build_gp(X_norm, y)

            next_norm, ei_val = suggest_next_point(
                self.gp, self.factors, y.max(), self.xi, rng=self.rng
            )

            next_raw = self._denormalize(next_norm.reshape(1, -1))[0]

            # Snap categorical factors
            for j, f in enumerate(self.factors):
                if f.kind == "categorical" and f.levels is not None:
                    levels = np.array(f.levels)
                    next_raw[j] = levels[np.argmin(np.abs(levels - next_raw[j]))]

            return self._raw_to_dict(next_raw), ei_val

    def tell(self, params: dict, accuracy: float, ei: Optional[float], phase: str):
        """Record an observation."""
        x_raw = np.array([params[f.name] for f in self.factors], dtype=float)
        self.X_raw.append(x_raw)
        self.y_observed.append(accuracy)
        self.ei_values.append(ei)
        self.phases.append(phase)

        best_so_far = max(self.y_observed)
        self._log_iteration(params, accuracy, ei, phase, best_so_far)
        self.iteration += 1

    def check_convergence(self) -> Tuple[bool, Optional[str]]:
        """Check stopping criteria.

        Criterion 1 — Accuracy convergence: the best accuracy has not improved
        by more than convergence_tol (default 1%) over the last
        convergence_window BO iterations.

        Criterion 2 — Experiment cap: total experiments (initial + BO) have
        reached max_iterations. This is checked in the run() loop directly.

        Returns (converged, reason) where reason is a descriptive string or None.
        """
        bo_indices = [i for i, p in enumerate(self.phases) if p == "BO"]
        if len(bo_indices) < self.convergence_window:
            return False, None

        # Best accuracy before the last `convergence_window` BO iterations
        older = bo_indices[:-self.convergence_window]
        if not older:
            # Not enough history before the window to compare against
            return False, None

        best_before_window = max(self.y_observed[i] for i in older)
        best_overall = max(self.y_observed)
        improvement = best_overall - best_before_window

        if improvement < self.convergence_tol:
            return True, (
                f"Accuracy converged: best improved only {improvement:.2f}% "
                f"(< {self.convergence_tol}% threshold) over last "
                f"{self.convergence_window} BO iterations"
            )
        return False, None

    def get_best(self) -> Tuple[dict, float]:
        """Return the best parameters and accuracy observed."""
        idx = int(np.argmax(self.y_observed))
        return self._raw_to_dict(self.X_raw[idx]), self.y_observed[idx]

    def plot_convergence(self, save_path: str = "doe_convergence.png"):
        """Plot accuracy vs iteration with best-so-far overlay."""
        iters = np.arange(len(self.y_observed))
        accuracies = np.array(self.y_observed)
        best_so_far = np.maximum.accumulate(accuracies)

        colors = ["#1f77b4" if p == "INITIAL" else "#ff7f0e" for p in self.phases]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        # Top: scatter of all observations
        ax1.scatter(iters, accuracies, c=colors, s=50, zorder=3, edgecolors="k", linewidths=0.5)
        ax1.axvline(x=self.n_initial - 0.5, color="gray", linestyle="--", alpha=0.7,
                     label="Initial → BO transition")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("DOE Optimization: Accuracy per Iteration")
        ax1.legend(["Phase boundary"], loc="lower right")
        # Custom legend for phases
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#1f77b4", edgecolor="k", label="Initial (LHS)"),
            Patch(facecolor="#ff7f0e", edgecolor="k", label="Bayesian Opt"),
        ]
        ax1.legend(handles=legend_elements, loc="lower right")
        ax1.grid(True, alpha=0.3)

        # Bottom: best-so-far line
        ax2.plot(iters, best_so_far, "r-o", markersize=4, linewidth=2, label="Best so far")
        ax2.axvline(x=self.n_initial - 0.5, color="gray", linestyle="--", alpha=0.7)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Best Accuracy")
        ax2.set_title("Cumulative Best Accuracy")
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\nConvergence plot saved to: {save_path}")
        plt.show()

    def run(self, simulator_fn: Callable = simulate_accuracy) -> Tuple[dict, float]:
        """Execute the full DOE optimization loop.

        Stopping criteria:
          1. Accuracy convergence — best accuracy improves < convergence_tol %
             over the last convergence_window BO iterations.
          2. Experiment cap — total experiments reach max_iterations (default 20).
        """
        stop_reason = None

        print("=" * 70)
        print("DOE OPTIMIZATION — Bayesian Optimization with Gaussian Process")
        print(f"Factors: {[f.name for f in self.factors]}")
        print(f"Initial design points: {self.n_initial}")
        print("Stopping criteria:")
        print(f"  1. Convergence: best accuracy improvement < {self.convergence_tol}% "
              f"over {self.convergence_window} consecutive BO iterations")
        print(f"  2. Experiment cap: {self.max_iterations} total experiments")
        print("=" * 70)

        # Open dashboard in browser
        dashboard_abs = os.path.abspath(self.dashboard.html_path)
        from urllib.parse import quote
        webbrowser.open(f"file://{quote(dashboard_abs)}")
        print(f"Dashboard opened: {dashboard_abs}")

        # Pre-experiment QC checks
        print(f"\n--- Pre-Experiment Quality Control ---")
        self.pre_experiment_qc = simulate_qc_checks_pre_experiment()
        for qc in self.pre_experiment_qc:
            status_str = "PASS" if qc.passed else "FAIL"
            print(f"  [{status_str}] {qc.name}: {qc.value}")
        self.dashboard.update(self)

        # Phase 1: Initial design
        print(f"\n--- Phase 1: Initial Design (LHS, {self.n_initial} points) ---")
        for i in range(self.n_initial):
            params, ei = self.suggest()
            accuracy = simulator_fn(**params)
            iter_qc = simulate_qc_checks_per_iteration(accuracy)
            self.per_iteration_qc.append(iter_qc)
            self.tell(params, accuracy, ei, "INITIAL")
            self.dashboard.update(self)

        # Phase 2: Bayesian optimization
        print(f"\n--- Phase 2: Bayesian Optimization ---")
        for i in range(self.n_initial, self.max_iterations):
            try:
                params, ei = self.suggest()
            except Exception as e:
                print(f"  [WARNING] GP suggestion failed ({e}), using random point")
                params = {}
                for f in self.factors:
                    if f.kind == "categorical":
                        params[f.name] = int(self.rng.choice(f.levels))
                    else:
                        params[f.name] = round(float(self.rng.uniform(f.low, f.high)), 2)
                ei = None

            accuracy = simulator_fn(**params)
            iter_qc = simulate_qc_checks_per_iteration(accuracy)
            self.per_iteration_qc.append(iter_qc)
            self.tell(params, accuracy, ei, "BO")

            # Criterion 1: accuracy convergence
            converged, reason = self.check_convergence()
            if converged:
                stop_reason = reason
                print(f"\n*** STOPPING (Criterion 1 — Convergence): {reason} ***")
                self.dashboard.update(self, stop_reason=stop_reason)
                break

            # Update dashboard — show the params we just tried as context
            self.dashboard.update(self, next_suggestion=params)
        else:
            # Criterion 2: experiment cap reached (loop completed without break)
            stop_reason = (
                f"Experiment cap reached: {self.max_iterations} experiments "
                f"completed without convergence"
            )
            print(f"\n*** STOPPING (Criterion 2 — Experiment Cap): {stop_reason} ***")
            self.dashboard.update(self, stop_reason=stop_reason)

        # Summary
        best_params, best_acc = self.get_best()
        print(f"\n{'=' * 70}")
        print(f"OPTIMIZATION COMPLETE — {self.iteration} total experiments")
        print(f"Stop reason: {stop_reason}")
        print(f"Best accuracy: {best_acc:.2f}")
        print(f"Best parameters: {best_params}")
        print(f"CSV log: {self.csv_path}")
        print(f"Dashboard: {dashboard_abs}")
        print(f"{'=' * 70}")

        self.plot_convergence()
        return best_params, best_acc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_doe(n_initial: int = 5, max_iterations: int = 20, xi: float = 0.01,
            convergence_tol: float = 1.0, convergence_window: int = 5,
            random_seed: int = 42) -> Tuple[dict, float]:
    """Run DOE optimization with default settings."""
    optimizer = DOEOptimizer(
        factors=FACTORS,
        n_initial=n_initial,
        max_iterations=max_iterations,
        xi=xi,
        convergence_tol=convergence_tol,
        convergence_window=convergence_window,
        random_seed=random_seed,
    )
    return optimizer.run(simulator_fn=simulate_accuracy)


if __name__ == "__main__":
    best_params, best_accuracy = run_doe()
