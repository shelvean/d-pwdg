"""Polish the M=1,...,10 finite-direction continuation results.

Author: Shelvean Kapita
Date: August 2026

This script does not alter the approximation space.  It restarts each converged
angle state with higher edge quadrature and a smaller finite-difference step so
that the reported recovery errors are not limited by the preliminary solve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from core.direction_adaptive import (
    l2_error,
    physical_skeleton_residual,
    physical_skeleton_residual_vector,
)
from experiments.experiment_finite_direction_capacity import (
    FinitePlaneSum,
    circular_match_error,
    state_for,
)

__author__ = "Shelvean Kapita"
__date__ = "August 2026"

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    input_data = pd.read_csv(RESULTS / "finite_direction_enrichment.csv")
    rows = []

    for _, record in input_data.iterrows():
        m = int(record.M)
        start = np.deg2rad(
            [float(x) for x in str(record.found_angles_deg).split(";")]
        )

        def residual(z):
            angles = np.mod(z, 2.0 * np.pi)
            state = state_for(m, angles, nq=20)
            return physical_skeleton_residual_vector(state)

        optimum = least_squares(
            residual,
            start,
            method="trf",
            x_scale=np.ones(m),
            ftol=1.0e-15,
            xtol=1.0e-15,
            gtol=1.0e-15,
            max_nfev=300,
            diff_step=2.0e-7,
            verbose=0,
        )

        angles = np.mod(optimum.x, 2.0 * np.pi)
        state = state_for(m, angles, nq=28)
        mean_error, max_error = circular_match_error(
            angles,
            FinitePlaneSum(m).angles,
        )

        row = {
            "M": m,
            "p": m,
            "dofs": state.ndof,
            "nfev_polish": optimum.nfev,
            "adaptive_L2": l2_error(state, nsub=18),
            "adaptive_J": physical_skeleton_residual(state) ** 2,
            "mean_angle_error_deg": mean_error,
            "max_angle_error_deg": max_error,
            "uniform_L2": record.uniform_L2,
            "found_angles_deg": ";".join(
                f"{x:.9f}" for x in np.sort(np.rad2deg(angles) % 360.0)
            ),
        }

        rows.append(row)
        print(row, flush=True)

    pd.DataFrame(rows).to_csv(
        RESULTS / "finite_direction_enrichment_polished.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
