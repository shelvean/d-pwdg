"""Top-level reproducibility launcher for the JSC Helmholtz manuscript.

Author: Shelvean Kapita
Date: August 2026

Examples
--------
python reproduce.py --list
python reproduce.py --quick
python reproduce.py --experiment transmission
python reproduce.py --experiment finite-directions
python reproduce.py --paper

`--paper` runs all fully rerunnable manuscript computations.  The hybrid adaptive and
DtN cutoff audits can be expensive.  High-precision trace-spectrum data reported in the
paper are supplied under data/audited; the original 50-digit generator was not part of
the surviving cleaned source tree and is therefore not falsely advertised as rerunnable.
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXPERIMENTS = {
    "crossing": [sys.executable, "-m", "experiments.experiment_crossing_waves"],
    "nonlinear-history": [sys.executable, "-m", "experiments.experiment_nonlinear_history"],
    "transmission": [sys.executable, "-m", "experiments.experiment_transmission"],
    "dtn-centered": [sys.executable, "-m", "experiments.experiment_dtn_direction_recovery", "--case", "centered", "--k", "8", "--nr", "1", "--ntheta", "8", "--p", "3"],
    "dtn-offcenter": [sys.executable, "-m", "experiments.experiment_dtn_direction_recovery", "--case", "offcenter", "--k", "8", "--nr", "1", "--ntheta", "8", "--p", "3"],
    "variable-projection": [sys.executable, "-m", "experiments.experiment_variable_projection_hankel"],
    "finite-capacity": [sys.executable, "-m", "experiments.experiment_finite_direction_capacity"],
    "finite-enrich-1-10": [sys.executable, "-m", "experiments.experiment_finite_direction_enrichment"],
    "finite-polish": [sys.executable, "-m", "experiments.polish_finite_direction_enrichment"],
    "finite-enrich-11-20": [sys.executable, "-m", "experiments.experiment_finite_direction_enrichment_11_20"],
    "hybrid": [sys.executable, "-m", "experiments.experiment_adaptive_threshold"],
    "dtn-tau-audit": [sys.executable, "-m", "experiments.precision_audit.experiment_dtn_tau_sweep"],
}

GROUPS = {
    "quick": ["crossing", "transmission"],
    "finite-directions": ["finite-capacity", "finite-enrich-1-10", "finite-polish", "finite-enrich-11-20"],
    "dtn-directions": ["dtn-centered", "dtn-offcenter"],
    "paper": ["crossing", "nonlinear-history", "transmission", "dtn-centered", "dtn-offcenter", "variable-projection", "finite-capacity", "finite-enrich-1-10", "finite-polish", "finite-enrich-11-20", "dtn-tau-audit", "hybrid"],
}

def run_one(name: str) -> None:
    cmd = EXPERIMENTS[name]
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(cmd), flush=True)
    tic = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"=== completed {name} in {time.time()-tic:.1f} s ===", flush=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--experiment", choices=sorted(set(EXPERIMENTS)|set(GROUPS)))
    args = ap.parse_args()
    if args.list:
        print("Experiments:")
        for x in EXPERIMENTS: print("  ", x)
        print("Groups:")
        for x, vals in GROUPS.items(): print(f"  {x}: {', '.join(vals)}")
        return
    if args.quick: names=GROUPS['quick']
    elif args.paper: names=GROUPS['paper']
    elif args.experiment:
        names=GROUPS.get(args.experiment,[args.experiment])
    else:
        ap.error("choose --list, --quick, --paper, or --experiment NAME")
    for name in names: run_one(name)

if __name__ == "__main__":
    main()
