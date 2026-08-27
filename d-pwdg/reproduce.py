"""Top-level reproducibility launcher for the JSC Helmholtz manuscript.

Author: Shelvean Kapita
Date: August 2026

This repository holds only what the submitted manuscript needs.  Each entry
below produces a numbered table or figure; nothing here is a development or
diagnostic script.

Examples
--------
python reproduce.py --list
python reproduce.py --quick
python reproduce.py --experiment table3
python reproduce.py --paper

Tables 1, 2, 3 and Fig. 1 come from `audited_source/`, the implementation that
produced the printed values, and use the analytic direction Jacobian described
in Section 7.1 of the paper.  The remaining entries come from `core/`, which is
the only implementation of the DtN and finite-direction experiments.

A failing step no longer aborts the run: the launcher records the
failure, skips only steps that consume its outputs, continues with
everything else, and prints a summary; the exit code is nonzero if
anything failed.  Pass ``--stop-on-fail`` for the old behavior.

`--paper` runs every rerunnable manuscript computation.  The hybrid benchmark
and the DtN cutoff audit are expensive.  The high-precision trace-spectrum data
reported in the paper are supplied under `data/audited/`; the original 50-digit
generator was not part of the surviving source tree and is therefore not
advertised here as rerunnable.
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDITED = ROOT / "audited_source"

# name -> (command, working directory, manuscript item)
EXPERIMENTS = {
    "table1": ([sys.executable, "uniform_conditioning_graph_riesz.py"], AUDITED,
               "Table 1 - uniform vs graph-Riesz conditioning"),
    "table2": ([sys.executable, "test_threewave_large_mesh.py"], AUDITED,
               "Table 2 - three hidden plane waves"),
    "table3": ([sys.executable, "verify_table3_exact.py"], ROOT,
               "Table 3 - transmission recovery (asserts the submitted rows)"),
    "fig1": ([sys.executable, "plot_transmission_visualization.py"], AUDITED,
             "Fig. 1 - transmission fields at omega=12"),
    "dtn-centered": ([sys.executable, "-m", "experiments.experiment_dtn_direction_recovery",
                      "--case", "centered", "--k", "8", "--nr", "1", "--ntheta", "8", "--p", "3"], ROOT,
                     "Table 4 / Fig. 4 - centered Hankel source"),
    "dtn-offcenter": ([sys.executable, "-m", "experiments.experiment_dtn_direction_recovery",
                       "--case", "offcenter", "--k", "8", "--nr", "1", "--ntheta", "8", "--p", "3"], ROOT,
                      "Table 4 / Fig. 4 - off-center Hankel source"),
    "dtn-tau-audit": ([sys.executable, "-m", "experiments.precision_audit.experiment_dtn_tau_sweep"], ROOT,
                      "Table 6 - trace-cutoff sweep"),
    "finite-capacity": ([sys.executable, "-m", "experiments.experiment_finite_direction_capacity"], ROOT,
                        "Table 7 - finite-direction capacity"),
    "finite-enrich-1-10": ([sys.executable, "-m", "experiments.experiment_finite_direction_enrichment"], ROOT,
                           "Fig. 7 - ENRICH-MOVE continuation, M=1..10"),
    "finite-polish": ([sys.executable, "-m", "experiments.polish_finite_direction_enrichment"], ROOT,
                      "Fig. 7 - continuation polish"),
    "finite-enrich-11-20": ([sys.executable, "-m", "experiments.experiment_finite_direction_enrichment_11_20"], ROOT,
                            "Fig. 7 - ENRICH-MOVE continuation, M=11..20"),
    "hybrid": ([sys.executable, "-m", "experiments.experiment_adaptive_threshold"], ROOT,
               "Table 8 / Fig. 8 - threshold-driven hybrid benchmark"),
    "variable-projection": ([sys.executable, "direct_hankel_multidir_joint.py"], AUDITED,
                            "Section 7.9 - variable-projection check"),
}

GROUPS = {
    "quick": ["table1", "table3", "fig1"],
    "tables-1-3": ["table1", "table2", "table3"],
    "finite-directions": ["finite-capacity", "finite-enrich-1-10", "finite-polish", "finite-enrich-11-20"],
    "dtn-directions": ["dtn-centered", "dtn-offcenter"],
    "paper": ["table1", "table2", "table3", "fig1", "dtn-centered", "dtn-offcenter",
              "variable-projection", "finite-capacity", "finite-enrich-1-10", "finite-polish",
              "finite-enrich-11-20", "dtn-tau-audit", "hybrid"],
}


# name -> names of earlier steps whose outputs it consumes (Fig. 7 chain)
DEPENDS = {
    "finite-polish": ["finite-enrich-1-10"],
    "finite-enrich-11-20": ["finite-enrich-1-10", "finite-polish"],
}


def run_one(name: str) -> tuple[str, float]:
    """Run one experiment; return (status, seconds), status in PASS/FAIL."""
    cmd, cwd, item = EXPERIMENTS[name]
    print(f"\n=== {name}  ({item}) ===", flush=True)
    print(" ".join(str(c) for c in cmd), flush=True)
    tic = time.time()
    proc = subprocess.run(cmd, cwd=cwd)
    dt = time.time() - tic
    if proc.returncode == 0:
        print(f"=== completed {name} in {dt:.1f} s ===", flush=True)
        return "PASS", dt
    print(f"=== FAILED {name} (exit {proc.returncode}) after {dt:.1f} s ===",
          flush=True)
    return "FAIL", dt


def run_all(names: list[str], stop_on_fail: bool) -> int:
    status: dict[str, str] = {}
    times: dict[str, float] = {}
    for name in names:
        broken = [d for d in DEPENDS.get(name, ())
                  if status.get(d) in ("FAIL", "SKIP")]
        if broken:
            print(f"\n=== SKIPPED {name}: depends on failed step(s) "
                  f"{', '.join(broken)} ===", flush=True)
            status[name] = "SKIP"
            continue
        status[name], times[name] = run_one(name)
        if status[name] == "FAIL" and stop_on_fail:
            break
    print("\n" + "=" * 60)
    print("Reproduction summary")
    for name in names:
        st = status.get(name, "NOT RUN")
        t = f"{times[name]:8.1f} s" if name in times else " " * 10
        print(f"  {st:7s} {t}  {name:22s} {EXPERIMENTS[name][2]}")
    n_fail = sum(v == "FAIL" for v in status.values())
    n_skip = sum(v == "SKIP" for v in status.values())
    n_pass = sum(v == "PASS" for v in status.values())
    print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped, "
          f"{len(names) - len(status)} not run")
    return 1 if (n_fail or n_skip) else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--experiment", choices=sorted(set(EXPERIMENTS) | set(GROUPS)))
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="abort at the first failing step (old behavior)")
    args = ap.parse_args()
    if args.list:
        print("Experiments:")
        for name, (_, _, item) in EXPERIMENTS.items():
            print(f"   {name:22s} {item}")
        print("Groups:")
        for name, vals in GROUPS.items():
            print(f"   {name}: {', '.join(vals)}")
        return
    if args.quick:
        names = GROUPS["quick"]
    elif args.paper:
        names = GROUPS["paper"]
    elif args.experiment:
        names = GROUPS.get(args.experiment, [args.experiment])
    else:
        ap.error("choose --list, --quick, --paper, or --experiment NAME")
    sys.exit(run_all(names, stop_on_fail=args.stop_on_fail))


if __name__ == "__main__":
    main()
