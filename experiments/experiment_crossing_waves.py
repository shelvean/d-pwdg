"""Crossing-wave direction recovery experiment.

Author
------
Shelvean Kapita
Department of Mathematics, Texas A&M University

Date
----
August 2026

Purpose
-------
This script reproduces the clean crossing-wave test used in the numerical
section of the manuscript.  The exact field is the sum of two plane waves with
angles 37 and 128 degrees.  The angles are used only to generate boundary data
and to measure the recovered directions after the nonlinear solve.

The experiment answers a simple question: how many local directions are needed
to represent two crossing wave fronts?  We solve the same problem with one,
two, and three global direction variables.  Each direction is available on
every element of the uniform eight-triangle mesh.

Run
---
From the top level of the code archive,

    python -m experiments.experiment_crossing_waves

The script writes ``results/two_plane_crossing.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from core.direction_adaptive import (
    PWDGDictionary,
    l2_error,
    physical_skeleton_residual,
    physical_skeleton_residual_vector,
    q_real,
)
from core.pwdg import square_mesh

__author__ = "Shelvean Kapita"
__date__ = "August 2026"

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

WAVENUMBER = 12.0
EXACT_ANGLES_DEG = (37.0, 128.0)
AMPLITUDES = (1.0, 0.8)
MESH_CELLS_PER_SIDE = 2  # 2 x 2 squares = 8 triangles.


class TwoPlaneWave:
    """Exact sum of two propagating plane waves."""

    def __init__(
        self,
        k: float = WAVENUMBER,
        angle1: float = np.deg2rad(EXACT_ANGLES_DEG[0]),
        angle2: float = np.deg2rad(EXACT_ANGLES_DEG[1]),
    ) -> None:
        self.k = float(k)
        self.angle1 = float(angle1)
        self.angle2 = float(angle2)

    def wavenumber(self, y):
        """Return the constant wavenumber with the shape of ``y``."""
        return self.k * np.ones_like(np.asarray(y, dtype=float))

    def __call__(self, x, y):
        """Evaluate the exact field at the supplied coordinates."""
        phase1 = self.k * (
            np.cos(self.angle1) * x + np.sin(self.angle1) * y
        )
        phase2 = self.k * (
            np.cos(self.angle2) * x + np.sin(self.angle2) * y
        )
        return AMPLITUDES[0] * np.exp(1j * phase1) + AMPLITUDES[1] * np.exp(
            1j * phase2
        )


def solve_with_angles(angles, nq:
    int = 16):
    """Solve PWDG with the same candidate directions on every element."""
    exact = TwoPlaneWave()
    vertices, triangles = square_mesh(MESH_CELLS_PER_SIDE)
    local_vectors = np.array([q_real(exact.k, angle) for angle in angles])
    dictionaries = [local_vectors.copy() for _ in range(len(triangles))]

    state = PWDGDictionary(
        MESH_CELLS_PER_SIDE,
        exact,
        dictionaries,
        nq=nq,
    )
    state.solve()
    return state


def residual_vector(angle_parameters):
    """Residual supplied to SciPy's trust-region least-squares solver."""
    try:
        angles = np.mod(np.asarray(angle_parameters, dtype=float), 2.0 * np.pi)
        state = solve_with_angles(angles, nq=10)
        return physical_skeleton_residual_vector(state)
    except Exception:
        # A fixed large vector lets the trust-region solver reject a trial point
        # without terminating the full experiment.
        return np.full(1000, 1.0e3)


def run_case(number_of_directions:
    int, initial_angles):
    """Optimize one crossing-wave case and return its diagnostics."""
    result = least_squares(
        residual_vector,
        np.asarray(initial_angles, dtype=float),
        method="trf",
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=120,
        diff_step=2.0e-5,
    )

    angles = np.mod(result.x, 2.0 * np.pi)
    state = solve_with_angles(angles, nq=20)

    return {
        "number_of_directions": number_of_directions,
        "nfev": result.nfev,
        "skeleton_residual_squared": physical_skeleton_residual(state) ** 2,
        "relative_L2_error": l2_error(state, nsub=14),
        "angles_deg": ";".join(f"{a:.8f}" for a in np.degrees(angles) % 360.0),
    }


def main() -> None:
    """Run the one-, two-, and three-direction comparisons."""
    starts = ([0.3], [0.3, 2.5], [0.3, 2.5, 4.8])
    rows = []

    print("Crossing-wave direction recovery")
    print(f"Exact angles: {EXACT_ANGLES_DEG[0]:.1f} and {EXACT_ANGLES_DEG[1]:.1f} degrees")
    for initial in starts:
        row = run_case(len(initial), initial)
        rows.append(row)
        print(row)

    output = RESULTS / "two_plane_crossing.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
