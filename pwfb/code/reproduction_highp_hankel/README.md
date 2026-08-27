# Exact reproduction package: high-p centered Hankel PW experiment

This package is the audited source for the high-order plane-wave experiment

\[
u(x)=H_0^{(1)}(16|x|)
\]

on the exact annulus \(0.5<r<1\), using eight congruent sectors, local trace
radius \(h=0.44\), and local centers at radius \(0.75\).

## One command

```bash
python reproduce_highp_hankel.py --out results
```

The run writes:

- `results/highp_hankel_reproduced.csv`
- `results/table_rows.tex`

Typical runtime is under one minute on a recent laptop. The independent
90-digit boundary-quadrature validation is the most expensive part.

## Environment

The code was audited with Python 3.13 and requires:

- numpy
- scipy
- pandas
- mpmath

Install the pinned versions in `requirements.txt` for the closest replication.

## What is computed

### Trace error

The exact centered Hankel field is expanded about the local disk center using
Graf's addition theorem.  The plane-wave trace fit is performed in the
Fourier-Bessel trace metric.  For equispaced directions, the least-squares
problem separates into residue classes modulo \(p\), so no ill-conditioned
normal equations are formed.

### Raw condition number

`raw_trace_gram_condition` is

\[
\kappa_{\rm tr,raw}
  = \lambda_{\max}(G_{\rm tr})/\lambda_{\min}(G_{\rm tr}),
\]

where \(G_{\rm tr}\) is the Cauchy-trace Gram matrix of the **raw equispaced
plane waves**.  The eigenvalues are evaluated from the exact circulant symbol.

### Condition number after trace-Riesz orthonormalization

`orth_trace_gram_condition_minus_1` records

\[
\kappa_{\rm tr,orth}-1.
\]

The trace-Riesz transformation is defined from the 70-digit modal Gram spectrum, but
the conditioned Gram matrix is **recomputed independently** from the physical
boundary integral using a 1024-point periodic quadrature at 90 decimal digits.
Thus the near-one values are a numerical validation, not a tautological reuse
of the same matrix.

This is a local trace-space condition number. It is **not**
\(\kappa_{\rm GR}\), the condition number of the globally graph-Riesz
normalized PWDG system.

### Broken L2 error

At high \(p\), direct evaluation of `u - u_h` in ordinary double precision
suffers catastrophic cancellation because both fields are O(1) while their
difference is O(1e-16).  The script therefore evaluates the *residual
Fourier-Bessel expansion directly* and integrates its square with 160x160
Gauss-Legendre quadrature.  Increasing the quadrature from 100 to 220 points
per coordinate leaves the reported digits unchanged.

## Important audit correction

During construction of this reproduction package, the high-p trace errors and
raw Gram condition numbers were reproduced exactly, but the earlier manuscript
L2 column was found to come from a less stable postprocessing calculation.
The residual-expansion evaluation is the auditable value and should be used in
the paper.

The independently recomputed orthogonalized condition numbers were also
re-run from scratch; the package outputs the audited values.

Do not edit the CSV by hand. Regenerate it with the script.
