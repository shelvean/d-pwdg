"""Audited DtN trace-cutoff sweep in IEEE double precision.

Author: Shelvean Kapita
Date: August 2026

The experiment uses the exact annulus 0.5 < r < 1 with one radial layer and
eight exact curved sectors, k=8, DtN truncation N=40, and 44-point edge and
trace quadrature.  It records how the retained trace rank and relative L2 error
change as the relative cutoff tau_rank is decreased.

The exact field is the plane-wave scattering solution for a sound-soft disk.
This is the same physical scattering benchmark used in the high-p DtN study.
"""

from __future__ import annotations

import csv
from pathlib import Path
import time

import numpy as np

from core.compressed_dtn import CompressedDtNPWDG
from core.dtn_pwdg import PolarAnnulusMesh, SoundSoftDiskScattering

__author__ = "Shelvean Kapita"
__date__ = "August 2026"

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "precision_audit"
RESULTS.mkdir(parents=True, exist_ok=True)

P_VALUES = [27, 29, 31, 33, 35]
TAU_VALUES = [
    1.0e-12,
    1.0e-14,
    1.0e-15,
    5.0e-16,
    2.0e-16,
    1.0e-16,
    5.0e-17,
    1.0e-17,
    0.0,
]


def main():
    mesh = PolarAnnulusMesh(0.5, 1.0, nr=1, ntheta=8)
    exact = SoundSoftDiskScattering(k=8.0, a=0.5, series_N=120)
    rows = []

    for p in P_VALUES:
        # Same uniform p-direction family on every curved sector.
        angles = np.tile(2.0 * np.pi * np.arange(p) / p, mesh.ne)

        for tau in TAU_VALUES:
            tic = time.time()
            solver = CompressedDtNPWDG(
                mesh,
                8.0,
                angles,
                exact,
                p,
                dtn_N=40,
                edge_order=44,
                trace_order=44,
                tau_rank=tau,
            )
            solver.assemble()
            solver.solve_graph_riesz(rtol=2.0e-13, maxiter=800)
            error = solver.relative_l2_error(order=20)

            relative_positive_eigenvalues = []
            nonpositive_count = 0
            for eigenvalues in solver.trace_eigs:
                nonpositive_count += int(np.sum(eigenvalues <= 0.0))
                positive = eigenvalues[eigenvalues > 0.0]
                if len(positive):
                    relative_positive_eigenvalues.append(
                        positive[0] / eigenvalues[-1]
                    )

            row = {
                "p": p,
                "tau": tau,
                "nominal": mesh.ne * p,
                "effective": solver.ndof,
                "rank_min": min(solver.ranks),
                "rank_max": max(solver.ranks),
                "rank_mean": float(np.mean(solver.ranks)),
                "rel_L2": error,
                "linear_residual": solver.linear_residual,
                "gmres_it": solver.gmres_iterations,
                "nonpositive_trace_eigs": nonpositive_count,
                "min_positive_rel_eig": (
                    float(min(relative_positive_eigenvalues))
                    if relative_positive_eigenvalues
                    else np.nan
                ),
                "seconds": time.time() - tic,
            }
            rows.append(row)
            print(row, flush=True)

            # Rewrite after every completed run so a long audit can be resumed
            # without losing the already completed parameter combinations.
            output = RESULTS / "tau_sweep_audited.csv"
            with output.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
