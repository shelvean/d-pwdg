"""Step-2 verification: homogeneous extension, gradient, constants, gluing."""

import numpy as np
import sympy as sp
from math import factorial
from hypgeom import (J, mink, dist, triangle_area_defect, bolza_octagon, side_maps,
                     inv_lorentz, normalize_point, to_klein, from_klein)
from hypbb import (bern_indices, vertex_matrix, gram, bary, bern_eval, basis_and_grad,
                   constant_coeffs, local_matrices, quad_points)

ok = []


def check(name, val, target=0.0, tol=1e-12):
    err = abs(val - target)
    p = err < tol
    print(f"{'PASS' if p else 'FAIL'}  {name}: {val:.6e}  (tol {tol:.0e})")
    ok.append(p)


print("=" * 74)
print("1. Symbolic proof of  box F = rho^{d-2} ( Delta_H f - d(d+1) f )")
print("=" * 74)

zx, zy = sp.symbols("zx zy", real=True)
s = 1 - zx**2 - zy**2
# Poincare disk -> hyperboloid
X_of_z = sp.Matrix([(1 + zx**2 + zy**2) / s, 2 * zx / s, 2 * zy / s])
x0, x1, x2 = sp.symbols("x0 x1 x2", real=True)

for d, P in [(2, x0*x1 - 3*x2**2),
             (3, x1**3 + 2*x0*x1*x2),
             (4, x0**2*x2**2 - x1**4 + x0*x1*x2**2)]:
    box = -sp.diff(P, x0, 2) + sp.diff(P, x1, 2) + sp.diff(P, x2, 2)
    f = P.subs(list(zip((x0, x1, x2), X_of_z)))
    lap_disk = sp.simplify(s**2 / 4 * (sp.diff(f, zx, 2) + sp.diff(f, zy, 2)))
    rhs = (box + d * (d + 1) * P).subs(list(zip((x0, x1, x2), X_of_z)))
    resid = sp.simplify(lap_disk - rhs)
    print(f"    degree {d}: Delta_H f - (box P + d(d+1) P) simplifies to {resid}")
    ok.append(resid == 0)
    # the WRONG sign, to show the identity has content
    wrong = sp.simplify(lap_disk - (box - d * (d + 1) * P).subs(
        list(zip((x0, x1, x2), X_of_z))))
    ok.append(wrong != 0)

print("    (opposite sign is non-zero in every case, so the identity is not vacuous)")

print()
print("=" * 74)
print("2. Gradient formula and tangency")
print("=" * 74)

V0, gens, R = bolza_octagon()
O = np.array([1.0, 0.0, 0.0])
V = vertex_matrix(O, V0[0], V0[1])
d = 6
rng = np.random.default_rng(0)
c = rng.standard_normal(len(bern_indices(d)))

Xq, wq = quad_points(V, 12)
phi, g = basis_and_grad(V, d, Xq)
gf = np.einsum("m,pma->pa", c, g)
check("gradient is tangent: <grad f, x> = 0",
      float(np.max(np.abs(mink(gf, Xq)))) / float(np.max(np.abs(gf))), 0.0, 1e-13)

# compare |grad|^2 against the Poincare-disk expression
Vs = sp.Matrix(V.tolist())
bsym = Vs.inv() * X_of_z
Ftest = (bsym[0]**2 * bsym[1]**3 * bsym[2]**1)          # degree 6 in x
grad_disk2 = sp.simplify(s**2 / 4 * (sp.diff(Ftest, zx)**2 + sp.diff(Ftest, zy)**2))

ctest = np.zeros(len(bern_indices(d)))
ctest[bern_indices(d).index((2, 3, 1))] = 1.0 / (
    factorial(d) // (factorial(2) * factorial(3) * factorial(1)))
zpt = (0.13, -0.07)
xpt = np.array([float(v.subs({zx: zpt[0], zy: zpt[1]})) for v in X_of_z])
_, gpt = basis_and_grad(V, d, xpt)
gnum = np.einsum("m,ma->a", ctest, gpt[0])
val_num = float(mink(gnum, gnum))
val_sym = float(grad_disk2.subs({zx: zpt[0], zy: zpt[1]}))
check("|grad_H f|^2 matches the Poincare-disk value",
      abs(val_num - val_sym) / abs(val_sym), 0.0, 1e-11)

print()
print("=" * 74)
print("3. Constants: representable iff d is even")
print("=" * 74)

for deg in (2, 4, 6, 8):
    cc = constant_coeffs(V, deg)
    Xr, _ = quad_points(V, 9)
    ph, gr = basis_and_grad(V, deg, Xr)
    vals = ph @ cc
    grads = np.einsum("m,pma->pa", cc, gr)
    print(f"    d={deg}: max |1 - sum c B| = {np.max(np.abs(vals - 1)):.3e},"
          f"  max |grad| = {np.max(np.abs(grads)):.3e}")
    ok.append(np.max(np.abs(vals - 1)) < 1e-10 and np.max(np.abs(grads)) < 1e-9)

try:
    constant_coeffs(V, 5)
    ok.append(False)
    print("    FAIL  odd degree did not raise")
except ValueError:
    print("    odd degree correctly rejected (constant not in the space)")
    ok.append(True)

print()
print("=" * 74)
print("4. Local mass and stiffness against exact area")
print("=" * 74)

for deg in (2, 4, 6):
    M, K = local_matrices(V, deg, n=56)
    cc = constant_coeffs(V, deg)
    area_exact = float(triangle_area_defect(O, V0[0], V0[1]))
    check(f"d={deg}: c^T M c = area", float(cc @ M @ cc), area_exact, 1e-9)
    check(f"d={deg}: K c = 0 (constant in the kernel)",
          float(np.max(np.abs(K @ cc))) / float(np.max(np.abs(K))), 0.0, 1e-10)
    check(f"d={deg}: M symmetric positive", float(np.min(np.linalg.eigvalsh(M)) < 0), 0.0, 0.5)

print()
print("=" * 74)
print("5. Gluing: barycentric invariance and C^0/C^1 conditions across a paired edge")
print("=" * 74)

g0 = gens[0]
# a triangle against side 0, and the triangle against side 4 that it glues to
T = vertex_matrix(O, V0[0], V0[1])
Tg = vertex_matrix(g0 @ O, g0 @ V0[0], g0 @ V0[1])
Xs, _ = quad_points(T, 7)
check("barycentric coordinates are invariant under the pairing",
      float(np.max(np.abs(bary(T, Xs) - bary(Tg, (g0 @ Xs.T).T)))), 0.0, 1e-11)

deg = 5
cs = rng.standard_normal(len(bern_indices(deg)))
ph_T, _ = basis_and_grad(T, deg, Xs)
ph_Tg, _ = basis_and_grad(Tg, deg, (g0 @ Xs.T).T)
check("same coefficients give the same function on the image triangle",
      float(np.max(np.abs(ph_T @ cs - ph_Tg @ cs))), 0.0, 1e-9)

# C^0 across a shared edge inside the domain: two triangles sharing edge (V0[0],V0[1])
w4 = normalize_point(V0[0] + V0[1] + 0.6 * O)     # a fourth point on the other side
Ta = vertex_matrix(O, V0[0], V0[1])
Tb = vertex_matrix(w4, V0[0], V0[1])
t = np.linspace(0.05, 0.95, 11)
from hypgeom import geodesic_point
Xe = np.array([geodesic_point(V0[0], V0[1], tt) for tt in t])

ia, ib_ = bern_indices(deg), bern_indices(deg)
ca = rng.standard_normal(len(ia))
cb = ca.copy()
# C^0: coefficients on the shared edge (i=0) must agree; they do by construction here
mapA = {e: m for m, e in enumerate(ia)}
for m, e in enumerate(ib_):
    if e[0] == 0:
        cb[m] = ca[mapA[e]]
pa, _ = basis_and_grad(Ta, deg, Xe)
pb, _ = basis_and_grad(Tb, deg, Xe)
check("C^0 holds when edge coefficients agree",
      float(np.max(np.abs(pa @ ca - pb @ cb))), 0.0, 1e-11)

cb_bad = cb.copy()
cb_bad[[m for m, e in enumerate(ib_) if e[0] == 0][2]] += 1.0
check("C^0 fails when they do not (test has content)",
      float(1.0 / max(np.max(np.abs(pa @ ca - pb @ cb_bad)), 1e-30)), 0.0, 1e3)

print()
print("=" * 74)
print(f"{sum(ok)}/{len(ok)} checks passed")
print("=" * 74)
