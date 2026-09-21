"""Step-1 diagnostics for the hyperbolic spline programme."""

import numpy as np
from hypgeom import (J, mink, dist, log_map, exp_map, frame, inv_lorentz, is_lorentz,
                     angle_at, triangle_area_defect, barycentric, from_barycentric,
                     to_klein, from_klein, klein_triangle_area, regular_polygon,
                     bolza_octagon, normalize_point, on_hyperboloid, pairing_map,
                     side_maps, vertex_cycle, cycle_word)

np.set_printoptions(precision=15, suppress=False)
ok = []


def check(name, val, target=None, tol=1e-12):
    if target is None:
        passed = bool(val)
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    else:
        err = abs(val - target)
        passed = err < tol
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {val!r}  (err {err:.3e})")
    ok.append(passed)


print("=" * 72)
print("1. Minkowski layer")
print("=" * 72)

V, R = regular_polygon(8, np.pi / 4)
check("vertices on hyperboloid", on_hyperboloid(V))
check("cosh R = cot^2(pi/8) = 3 + 2 sqrt(2)", np.cosh(R), 3 + 2 * np.sqrt(2))
r_poincare2 = (np.cosh(R) - 1) / (np.cosh(R) + 1)
check("Poincare radius^2 = 1/sqrt(2)", r_poincare2, 1 / np.sqrt(2))

p, q = V[0], V[3]
check("exp(log) round trip", float(dist(exp_map(p, log_map(p, q)), q)), 0.0, 1e-13)
check("exp(log) round trip, relative coordinates",
      float(np.max(np.abs(exp_map(p, log_map(p, q)) - q)) / np.max(np.abs(q))), 0.0, 1e-12)
check("|log_map| = dist", float(np.sqrt(mink(log_map(p, q), log_map(p, q)))),
      float(dist(p, q)), 1e-13)

F = frame(p, q)
check("frame in O(2,1): F^T J F = J", float(np.max(np.abs(F.T @ J @ F - J))), 0.0, 1e-13)
check("inv_lorentz exact", float(np.max(np.abs(inv_lorentz(F) @ F - np.eye(3)))), 0.0, 1e-13)

# barycentric linearity
Vt = np.column_stack([V[0], V[1], V[2]])
x = from_barycentric(Vt, [0.2, 0.5, 0.3])
b = barycentric(Vt, x)
check("barycentric round trip (up to scale)",
      float(np.max(np.abs(b / b.sum() - np.array([0.2, 0.5, 0.3]) / 1.0))), 0.0, 1e-12)
x1 = from_barycentric(Vt, [0.1, 0.6, 0.3])
lam = 0.37
b_mix = barycentric(Vt, lam * x + (1 - lam) * x1)
check("barycentric map is linear in x",
      float(np.max(np.abs(b_mix - (lam * barycentric(Vt, x)
                                   + (1 - lam) * barycentric(Vt, x1))))), 0.0, 1e-13)

print()
print("=" * 72)
print("2. Klein model quadrature vs exact angle defect")
print("=" * 72)
check("Klein round trip", float(np.max(np.abs(from_klein(to_klein(V)) - V))), 0.0, 1e-13)

O = np.array([1.0, 0.0, 0.0])
tri = (O, V[0], V[1])
exact = triangle_area_defect(*tri)
for n in (4, 8, 16, 32):
    approx = klein_triangle_area(*tri, n=n)
    print(f"    n={n:3d}  quad area {approx:.15f}   defect {exact:.15f}   "
          f"err {abs(approx-exact):.3e}")
check("Klein quadrature matches angle defect", klein_triangle_area(*tri, n=64),
      float(exact), 1e-12)

print()
print("=" * 72)
print("3. Bolza octagon: exact topological diagnostics")
print("=" * 72)
V, gens, R = bolza_octagon()

# area by angle defect over the 8 central triangles
area_defect = sum(triangle_area_defect(O, V[j], V[(j + 1) % 8]) for j in range(8))
check("area = 4 pi (g-1) with g=2, by angle defect", float(area_defect), 4 * np.pi, 1e-12)

area_quad = sum(klein_triangle_area(O, V[j], V[(j + 1) % 8], n=64) for j in range(8))
check("area = 4 pi by Klein quadrature", float(area_quad), 4 * np.pi, 1e-11)

angles = [angle_at(V[j], V[(j - 1) % 8], V[(j + 1) % 8]) for j in range(8)]
check("each interior angle = pi/4", float(np.max(np.abs(np.array(angles) - np.pi / 4))),
      0.0, 1e-13)
check("angle sum = 2 pi (vertex cycle closes)", float(np.sum(angles)), 2 * np.pi, 1e-13)

print()
print("    generators:")
for j, g in enumerate(gens):
    good, orth, det, a00 = is_lorentz(g)
    print(f"    g{j}: |g^T J g - J| = {orth:.3e}   det = {det:.15f}   g00 = {a00:.6f}")
    ok.append(good and abs(det - 1) < 1e-10 and a00 > 0)

# side pairing acts correctly on endpoints
err = 0.0
for j, g in enumerate(gens):
    err = max(err, float(np.max(np.abs(g @ V[j] - V[(j + 5) % 8]))))
    err = max(err, float(np.max(np.abs(g @ V[(j + 1) % 8] - V[(j + 4) % 8]))))
check("g_j maps side j onto side j+4 (endpoints)", err, 0.0, 1e-12)

# side lengths equal (needed for the pairing to be an isometry of the sides)
L = [float(dist(V[j], V[(j + 1) % 8])) for j in range(8)]
check("all 8 sides equal length", float(np.max(L) - np.min(L)), 0.0, 1e-13)
print(f"    side length = {L[0]:.15f},  cosh = {np.cosh(L[0]):.15f}")

print()
print("=" * 72)
print("4. Fuchsian surface relation and vertex orbit")
print("=" * 72)
g = side_maps(gens)
print(f"    vertex cycle (corner indices) = {vertex_cycle()}")
print("    relation  g0 g1^-1 g2 g3^-1 g0^-1 g1 g2^-1 g3 = I")
W = cycle_word(gens)
check("vertex cycle word = I  (vertex compatibility closes)",
      float(np.max(np.abs(W - np.eye(3)))), 0.0, 1e-9)
tr = [float(np.trace(m)) for m in gens]
print(f"    traces = {[round(t, 10) for t in tr]}   (|tr| > 3: hyperbolic translations)")
ok.append(all(abs(t) > 3 for t in tr))
ell = [float(np.arccosh(0.5 * (t - 1))) for t in tr]
print(f"    translation lengths = {[round(e, 12) for e in ell]}")
check("all four generators have equal translation length",
      float(max(ell) - min(ell)), 0.0, 1e-10)
check("translation length = Bolza systole = 2 arccosh(1 + sqrt 2)",
      float(ell[0]), float(2 * np.arccosh(1 + np.sqrt(2))), 1e-12)
check("2R = Fenchel-Nielsen l_1 = 2 arccosh(3 + 2 sqrt 2)",
      float(2 * R), float(2 * np.arccosh(3 + 2 * np.sqrt(2))), 1e-13)
print("    reference for the spectral step: lambda_1(Bolza) = 3.8388872588421995,"
      " multiplicity 3 (Strohmaier-Uski)")

# vertex orbit: all 8 vertices in one class
full = side_maps(gens)
orbit = [0]
frontier = [0]
while frontier:
    nxt = []
    for i in frontier:
        for gg in full:
            w = gg @ V[i]
            for k in range(8):
                if np.max(np.abs(w - V[k])) < 1e-9 and k not in orbit:
                    orbit.append(k)
                    nxt.append(k)
    frontier = nxt
check("all 8 vertices in a single orbit (one point on the surface)", len(orbit) == 8)
print(f"    orbit = {sorted(orbit)}")

print()
print("=" * 72)
print(f"{sum(ok)}/{len(ok)} checks passed")
print("=" * 72)
