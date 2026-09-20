# JSC manuscript source and reproducibility package

Manuscript title: **A High-Order C^0 Bernstein Quasi-Trefftz Method for Stationary Hamilton--Jacobi Equations**

This package contains the latest 28-page manuscript source together with the numerical code and data used in the manuscript.

## Contents

- `manuscript/main.tex`: JSC-ready LaTeX source.
- `manuscript/figures/`: manuscript figures.
- `manuscript/Kapita_Bernstein_QuasiTrefftz_Hamilton_Jacobi_JSC_ready.pdf`: PDF compiled from the included source.
- `reproducibility/code/`: Python implementation and benchmark/plot drivers.
- `reproducibility/data/`: CSV/NPZ data accompanying the numerical experiments.
- `reproducibility/README.md` and `README_EMBEDDED_QT.md`: code instructions and notes.

## JSC administrative changes included

- The manuscript title is exactly: `A High-Order C^0 Bernstein Quasi-Trefftz Method for Stationary Hamilton--Jacobi Equations`.
- Corresponding-author email: `kapita@tamu.edu`.
- Funding statement: `The author received no specific funding for this work.`

The scientific content is the latest uploaded manuscript. One malformed control byte in the archived LaTeX source before a `begin{remark}` command was repaired so the source compiles cleanly.

The Marmousi velocity arrays are not redistributed, consistent with the manuscript and the original code package; the relevant driver accepts externally supplied velocity files.
