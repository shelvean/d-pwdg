"""The Bolza rows of Table 'tab:gb': angle-defect area and Klein-chart
quadrature of the area, and the corner-cycle angle sum."""
import numpy as np, hypgeom as hg
V0, g, R = hg.bolza_octagon(); O = np.array([1., 0, 0])
A = sum(hg.triangle_area_defect(O, V0[j], V0[(j+1) % 8]) for j in range(8))
print('area by angle defect', A, 'defect', abs(A-4*np.pi))
for nq in [8, 16, 32, 48]:
    Aq = sum(hg.klein_triangle_area(O, V0[j], V0[(j+1) % 8], nq) for j in range(8)); print('Klein quadrature nq', nq, abs(Aq-4*np.pi))
print('corner cycle angle sum / pi =', sum(hg.angle_at(V0[j], V0[(j-1) % 8], V0[(j+1) % 8]) for j in range(8))/np.pi)
