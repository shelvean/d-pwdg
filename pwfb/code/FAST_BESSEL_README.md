# Fast Fourier--Bessel evaluation

The frozen manuscript driver `run_hybrid_global_dtn_esprit_stable.py` is left unchanged.

Two faster consecutive-order kernels are provided in `fast_bessel.py` and used by
`run_hybrid_global_dtn_esprit_stable_fast.py`.

## Miller mode (default)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python run_hybrid_global_dtn_esprit_stable_fast.py
```

This uses a vectorized continued-ratio form of Miller backward recurrence.  It evaluates
all positive integer orders together, reconstructs negative orders from
`J_-m=(-1)^m J_m`, and computes derivatives from
`J'_m=(J_{m-1}-J_{m+1})/2`.

On the supplied machine the full seven-budget hybrid run took 5.86 s versus 21.19 s for
the frozen reference driver (3.6x wall-clock speedup).  The selected PW/FB spaces and
retained ranks were unchanged.  At the final ~1e-13 error floor, roundoff-sensitive
quantities move slightly because the unscaled trace Gramians reach extreme condition
numbers.

## Conservative AMOS-block mode

```bash
BESSEL_MODE=amos OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python run_hybrid_global_dtn_esprit_stable_fast.py
```

This evaluates only nonnegative orders with SciPy/AMOS in one block and eliminates all
`jvp` calls.  Negative orders and derivatives are then algebraic.  The kernel is roughly
5--6x faster than separate `jv+jvp` evaluation while staying as close as possible to the
reference AMOS values.  Use this mode when reproducing roundoff-level table entries is
more important than maximum speed.

## Kernel benchmark

Run

```bash
python benchmark_fast_bessel.py
```

The Miller kernel was about 45--90x faster than separate `jv+jvp` calls for 4096 arguments
and M=10--60 in the benchmark, with max absolute discrepancies around 1.5e-13.
