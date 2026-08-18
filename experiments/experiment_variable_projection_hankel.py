"""Part B variable-projection experiment for an outgoing Hankel wave.

Author
------
Shelvean Kapita
Department of Mathematics, Texas A&M University

Date
----
August 2026

Purpose
-------
This script gives a transparent implementation of the one-ray Part B experiment
from the manuscript.  The PWDG state equation is not used.  For every set of
local angles, the linear Trefftz coefficients are obtained by least squares and
eliminated from the nonlinear problem.  The optimizer therefore sees only the
angle variables.

The supplied ``data/hankel_direct_joint.csv`` contains the reference results
reported in the manuscript.  The experiment below can be rerun from several
initial states to illustrate the same initialization sensitivity.

Run
---

    python -m experiments.experiment_variable_projection_hankel

The script writes ``results/variable_projection_hankel.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import hankel1
from scipy.stats import qmc

from core.direction_adaptive import PWDGDictionary, l2_error, q_complex_angle, q_real
from core.pwdg import square_mesh
from core.variable_projection import projected_state, real_projected_residual

__author__ = "Shelvean Kapita"
__date__ = "August 2026"

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

KAPPA = 8.0
SOURCE = np.array([-1.6, 0.3])
ETA_MAX = 1.8


class OutgoingHankel:
    """Exact cylindrical field with a source outside the computational square."""

    def __init__(self, k:
        float = KAPPA):
        self.k = float(k)

    def wavenumber(self, y):
        return self.k * np.ones_like(np.asarray(y, dtype=float))

    def __call__(self, x, y):
        radius = np.hypot(
            np.asarray(x) - SOURCE[0],
            np.asarray(y) - SOURCE[1],
        )
        return hankel1(0, self.k * radius)


def mesh_and_radial_angles():
    """Return the eight-triangle mesh and centroid-radial initial angles."""
    vertices, triangles = square_mesh(2)
    centroids = vertices[triangles].mean(axis=1)
    radial = np.arctan2(
        centroids[:, 1] - SOURCE[1],
        centroids[:, 0] - SOURCE[0],
    )
    return vertices, triangles, radial


def build_real_state(angles, nq:
    int = 16):
    exact = OutgoingHankel()
    _, triangles, _ = mesh_and_radial_angles()
    dictionaries = [
        np.asarray([q_real(KAPPA, angle)], dtype=complex) for angle in angles
    ]
    return PWDGDictionary(2, exact, dictionaries, nq=nq)


def build_complex_state(parameters, nq:
    int = 16):
    exact = OutgoingHankel()
    _, triangles, _ = mesh_and_radial_angles()
    z = np.asarray(parameters, dtype=float).reshape(len(triangles), 2)
    dictionaries = [
        np.asarray([q_complex_angle(KAPPA, theta, eta)], dtype=complex)
        for theta, eta in z
    ]
    return PWDGDictionary(2, exact, dictionaries, nq=nq)


def optimize_real(initial_angles, max_nfev:
    int = 180):
    """Optimize one real angle per element after eliminating coefficients."""

    def residual(x):
        angles = np.mod(np.asarray(x, dtype=float), 2.0 * np.pi)
        try:
            return real_projected_residual(build_real_state(angles, nq=12))
        except Exception:
            return np.full(2000, 1.0e3)

    result = least_squares(
        residual,
        np.asarray(initial_angles, dtype=float),
        method="trf",
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=max_nfev,
        diff_step=2.0e-5,
    )
    angles = np.mod(result.x, 2.0 * np.pi)
    state, reduced_residual, rank = projected_state(build_real_state(angles, nq=20))
    return result, state, reduced_residual, rank, angles


def optimize_complex(initial_angles, max_nfev:
    int = 180):
    """Optimize one complex angle ``theta+i eta`` per element."""
    initial = np.column_stack((initial_angles, np.zeros_like(initial_angles))).ravel()
    lower = np.tile([-np.inf, -ETA_MAX], len(initial_angles))
    upper = np.tile([np.inf, ETA_MAX], len(initial_angles))

    def residual(x):
        z = np.asarray(x, dtype=float).reshape(-1, 2).copy()
        z[:, 0] = np.mod(z[:, 0], 2.0 * np.pi)
        try:
            return real_projected_residual(build_complex_state(z.ravel(), nq=12))
        except Exception:
            return np.full(2000, 1.0e3)

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=max_nfev,
        diff_step=2.0e-5,
    )
    z = result.x.reshape(-1, 2)
    z[:, 0] = np.mod(z[:, 0], 2.0 * np.pi)
    state, reduced_residual, rank = projected_state(build_complex_state(z.ravel(), nq=20))
    return result, state, reduced_residual, rank, z


def summarize(label, model, result, state, residual, rank, directions):
    return {
        "initialization": label,
        "direction_model": model,
        "nfev": result.nfev,
        "J": float(np.vdot(residual, residual).real),
        "relative_L2": l2_error(state, nsub=14),
        "least_squares_rank": rank,
        "directions": np.array2string(np.asarray(directions), precision=8),
    }


def main() -> None:
    _, triangles, radial = mesh_and_radial_angles()

    # A deterministic perturbation is used so the calculation is reproducible.
    perturbation = np.deg2rad([2.0, -2.0, 1.5, -1.5, 2.5, -2.5, 1.0, -1.0])

    # Deterministic Sobol initialization on [0,2pi).
    sobol = qmc.Sobol(d=1, scramble=False)
    blind = 2.0 * np.pi * sobol.random_base2(m=3).ravel()[: len(triangles)]

    rows = []
    for label, initial in (
        ("centroid radial", radial),
        ("perturbed radial", radial + perturbation),
        ("blind Sobol", blind),
    ):
        result, state, residual, rank, angles = optimize_real(initial)
        row = summarize(label, "real", result, state, residual, rank, np.degrees(angles))
        rows.append(row)
        print(row)

    result, state, residual, rank, z = optimize_complex(radial)
    complex_directions = np.column_stack((np.degrees(z[:, 0]), z[:, 1]))
    row = summarize(
        "centroid radial",
        "complex",
        result,
        state,
        residual,
        rank,
        complex_directions,
    )
    rows.append(row)
    print(row)

    output = RESULTS / "variable_projection_hankel.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
