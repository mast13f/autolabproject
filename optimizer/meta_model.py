"""
MetaModel — Cross-Run Transfer Learning
========================================
Fits per-parameter Gaussian Processes across multiple completed runs
to predict optimal pipetting parameters for a new liquid + concentration
combination.

Usage (standalone — not called by the campaign runner)::

    from optimizer.meta_model import MetaModel

    meta = MetaModel()
    meta.fit("results/training_data.csv")
    meta.predict("ethanol", 100.0)
    meta.coverage_report("results/training_data.csv")

Requires ``pandas`` (lazy-imported so the main runner never depends on it).
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Optional

import numpy as np

from optimizer.core import (
    KNOWN_LIQUIDS,
    PIPETTING_FACTORS,
    TRAINING_CSV,
    build_gp,
)


class MetaModel:
    """Predicts optimal pipetting parameters from historical training data.

    Fits one GP per parameter using ``(liquid_type_encoded, concentration_pct)``
    as features and the best-per-run parameter value as the target.
    """

    def __init__(self):
        from sklearn.preprocessing import LabelEncoder

        self._models: Dict[str, object] = {}
        self._le = LabelEncoder()
        self._le.classes_ = np.array(KNOWN_LIQUIDS)
        self._trained = False
        self._factor_names = [f.name for f in PIPETTING_FACTORS]

    def fit(self, training_csv: str = TRAINING_CSV, min_runs: int = 3) -> bool:
        """Build cross-run GP models.

        Returns ``True`` if fitting succeeded, ``False`` if there is
        insufficient data (< *min_runs* distinct run IDs).
        """
        import pandas as pd

        if not os.path.exists(training_csv):
            print(f"[MetaModel] No training data found at {training_csv}")
            return False
        df = pd.read_csv(training_csv)
        if df.empty:
            print("[MetaModel] Training CSV is empty.")
            return False

        best_per_run = (
            df.sort_values("score", ascending=False)
            .groupby("run_id", as_index=False)
            .first()
        )
        n = len(best_per_run)
        if n < min_runs:
            print(
                f"[MetaModel] Only {n} run(s). Need at least {min_runs}. "
                f"Keep collecting data."
            )
            return False

        X = np.column_stack([
            self._le.transform(best_per_run["liquid_type"].str.lower()),
            best_per_run["concentration_pct"].values.astype(float),
        ])

        print(f"\n[MetaModel] Fitting on {n} run(s)...")
        for name in self._factor_names:
            if name not in best_per_run.columns:
                continue
            self._models[name] = build_gp(X, best_per_run[name].values.astype(float))
            print(f"  + {name}")

        self._trained = True
        print("[MetaModel] Ready.\n")
        return True

    def predict(
        self, liquid_type: str, concentration_pct: float,
    ) -> Optional[Dict[str, float]]:
        """Predict optimal parameters for a new liquid + concentration.

        Returns a dict of ``{factor_name: predicted_value}`` or ``None``
        if the model has not been fitted.
        """
        if not self._trained:
            print("[MetaModel] Not fitted yet. Call fit() first.")
            return None
        liquid_type = liquid_type.lower().strip()
        if liquid_type not in self._le.classes_:
            print(f"[MetaModel] Unknown liquid '{liquid_type}'.")
            return None

        X_new = np.array([
            [self._le.transform([liquid_type])[0], concentration_pct],
        ])
        print(
            f"\n[MetaModel] Predicted optimal params for "
            f"{liquid_type} @ {concentration_pct:.1f}%:"
        )
        predictions: Dict[str, float] = {}
        for name, gp in self._models.items():
            mu, sigma = gp.predict(X_new, return_std=True)
            f = next(fa for fa in PIPETTING_FACTORS if fa.name == name)
            val = float(np.clip(mu[0], f.low, f.high))
            if f.kind == "categorical" and f.levels:
                lvls = np.array(f.levels)
                val = float(lvls[np.argmin(np.abs(lvls - val))])
            predictions[name] = val
            print(f"  {name:20s} = {val:.2f}  (±{sigma[0]:.2f})")

        return predictions

    def coverage_report(self, training_csv: str = TRAINING_CSV):
        """Print a summary of training data coverage by liquid + concentration."""
        import pandas as pd

        if not os.path.exists(training_csv):
            print("No training data yet.")
            return
        df = pd.read_csv(training_csv)
        print("\n[MetaModel] Training data coverage:")
        summary = (
            df.groupby(["liquid_type", "concentration_pct"])
            .agg(
                n_iters=("iteration", "count"),
                best_score=("score", "max"),
                n_runs=("run_id", "nunique"),
            )
            .reset_index()
        )
        print(summary.to_string(index=False))
