"""Automatic propagation and evanescence in a two-material transmission problem.

Author
------
Shelvean Kapita
Department of Mathematics, Texas A&M University

Date
----
August 2026

This driver uses the same complex-angle parameterization for propagating and
evanescent states.  Two directions are active in the lower material and one in
the upper material.  Frequency continuation is used because the nonlinear
problem at the target frequency is not globally convex.

Run
---

    python -m experiments.experiment_transmission

The target frequency is omega=12.  The script runs the 69-degree propagating
case and the 29-degree total-internal-reflection case and writes a CSV summary
to ``results/transmission_recovery.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.direction_adaptive import (
    graph_data,
    l2_error,
    optimize_region_directions_ls,
    physical_skeleton_residual,
)
from core.exact_fields import TransmissionField

__author__ = "Shelvean Kapita"
__date__ = "August 2026"

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

CONTINUATION_FREQUENCIES = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)


def continuation_run(incidence_deg:
    float, mesh_n: int = 2):
    """Continue the three complex-angle variables to omega=12."""
    # Generic starting directions.  They are not computed from Snell's law.
    lower_z = np.array(
        [
            [np.deg2rad(55.0), 0.0],
            [np.deg2rad(305.0), 0.0],
        ],
        dtype=float,
    )
    upper_z = np.array([[np.deg2rad(25.0), 0.0]], dtype=float)

    cumulative_nfev = 0
    state = None
    for omega in CONTINUATION_FREQUENCIES:
        exact = TransmissionField(omega=omega, incidence_deg=incidence_deg)
        z, state, result = optimize_region_directions_ls(
            mesh_n,
            exact,
            lower_z,
            upper_z,
            nq=12,
            eta_max=1.8,
            max_nfev=250,
            verbose=False,
        )
        lower_z = z[:2]
        upper_z = z[2:]
        cumulative_nfev += result.nfev

    target = TransmissionField(omega=12.0, incidence_deg=incidence_deg)
    exact_lower, exact_upper = target.exact_complex_angles()
    graph = graph_data(state.A)

    return {
        "incidence_deg": incidence_deg,
        "mesh_triangles": int(state.nt),
        "coefficients": int(state.ndof),
        "cumulative_nfev": cumulative_nfev,
        "relative_L2": l2_error(state, nsub=14),
        "skeleton_residual": physical_skeleton_residual(state),
        "raw_condition": graph["cond_raw"],
        "graph_riesz_condition": graph["cond_gr"],
        "lower_theta_deg": ";".join(f"{x:.10f}" for x in np.degrees(lower_z[:, 0]) % 360.0),
        "lower_eta": ";".join(f"{x:.12e}" for x in lower_z[:, 1]),
        "upper_theta_deg": f"{np.degrees(upper_z[0,0]) % 360.0:.10f}",
        "upper_eta": f"{upper_z[0,1]:.12f}",
        "analytic_upper_theta_deg": f"{np.degrees(exact_upper[0,0]) % 360.0:.10f}",
        "analytic_upper_eta": f"{exact_upper[0,1]:.12f}",
    }


def main() -> None:
    rows = []
    for incidence in (69.0, 29.0):
        row = continuation_run(incidence)
        rows.append(row)
        print(row)

    output = RESULTS / "transmission_recovery.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
