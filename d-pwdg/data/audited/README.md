# Audited DtN and finite-direction results for manuscript use

This package intentionally contains only results that were actually executed
and subsequently audited. It excludes the provisional extrapolated
high-precision global compressed sweep.

Recommended manuscript use
--------------------------
1. DtN/Hankel limitation:
   Use the exact-circle Hankel examples to show that local phase directions can
   be identified accurately while a small plane-wave set does not represent
   a genuinely broad/cylindrical directional spectrum with comparable accuracy.

2. Arithmetic-precision barrier:
   Use dtn_precision_audited.csv / .png.
   The exact 8-sector annulus calculation compares raw double, graph-Riesz /
   trace-compressed double, 80-bit extended arithmetic, and the actually
   computed binary128 points. Do not interpolate missing binary128 points.

3. Compression-threshold sensitivity:
   Use tau_precision_audit.csv and p29_tau_sweep.png.
   Lowering tau_rank below 1e-12 retains useful modes and reduces error, but
   ordinary double precision eventually corrupts the smallest trace eigenvalues.

4. High-precision trace spectrum:
   Use high_precision_trace_spectrum.csv and trace_rank_recovery_50digit.png.
   The local trace Gramian and eigendecomposition were actually computed at
   50 decimal digits for p=29,31,35. All eigenvalues were positive there.
   This establishes that the nonpositive eigenvalues seen in double precision
   are numerical artifacts.

5. Finite-direction capacity:
   Use finite_direction_continuation_audited.csv / .png.
   The entire M=1,...,20 experiment is IEEE double precision. Automatic
   ENRICH-MOVE recovers M<=19 to essentially roundoff. At M=20 the automatic
   birth falls into a false basin, while an M=20 good-birth control, still in
   double precision, again reaches roundoff. Thus the M=20 transition is a
   nonlinear identification/birth issue, not an arithmetic-precision floor.

Do not use
----------
Do not use the provisional file
Kapita_DtN_Full_HighPrecision_Compressed_Study.zip
or its p=23,...,39 high-precision compressed curve as numerical evidence.
Those global high-precision compressed values were not fully executed.

Scientific stopping point
-------------------------
The current audited evidence is sufficient for the present manuscript:
- low directional complexity can be recovered essentially exactly;
- broad Hankel-type fields expose the limitation of sparse directional spaces;
- raw high-p PWDG has a finite-precision conditioning barrier;
- graph-Riesz normalization postpones that barrier;
- lowering tau_rank recovers additional useful trace modes;
- high-precision trace eigensolves show that further apparent rank loss in
  double precision is arithmetic, not structural.

A full arbitrary-precision compressed global solve is a useful future
verification experiment, but is not required to support these claims.
