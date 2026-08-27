"""Run manuscript-facing experiments using the audited source where available."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
AUD=ROOT/'audited_source'

def run(cmd,cwd):
    print('RUN:', ' '.join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd,cwd=cwd,check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('experiment',choices=[
        'transmission-and-conditioning','table3-verify','threewave-large-mesh',
        'finite-directions','dtn-directions','dtn-tau-audit','hybrid','variable-projection'])
    a=ap.parse_args()
    if a.experiment=='transmission-and-conditioning':
        run([sys.executable,'transmission_omega12_experiments.py'],AUD)
    elif a.experiment=='table3-verify':
        run([sys.executable,'verify_table3_exact.py'],ROOT)
    elif a.experiment=='threewave-large-mesh':
        run([sys.executable,'test_threewave_large_mesh.py'],AUD)
    elif a.experiment=='finite-directions':
        for mod in ['experiments.experiment_finite_direction_capacity','experiments.experiment_finite_direction_enrichment','experiments.polish_finite_direction_enrichment','experiments.experiment_finite_direction_enrichment_11_20']:
            run([sys.executable,'-m',mod],ROOT)
    elif a.experiment=='dtn-directions':
        for case in ['centered','offcenter']:
            run([sys.executable,'-m','experiments.experiment_dtn_direction_recovery','--case',case,'--k','8','--nr','1','--ntheta','8','--p','3'],ROOT)
    elif a.experiment=='dtn-tau-audit':
        run([sys.executable,'-m','experiments.precision_audit.experiment_dtn_tau_sweep'],ROOT)
    elif a.experiment=='hybrid':
        run([sys.executable,'-m','experiments.experiment_adaptive_threshold'],ROOT)
    elif a.experiment=='variable-projection':
        run([sys.executable,'direct_hankel_multidir_joint.py'],AUD)
if __name__=='__main__': main()
