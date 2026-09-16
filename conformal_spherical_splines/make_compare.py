"""make_compare.py: figure (L2(M) and H1(M) error against degrees of freedom,
spline d=2,4,6 vs isoparametric surface FEM P1,P2,P3) and a compact table
from compare_fem.log.  Writes fig_compare.pdf and tab_compare.tex into the
paper directories (Springer and SIAM)."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex': False, 'mathtext.fontset': 'cm', 'font.family': 'serif',
                     'font.serif': ['cmr10', 'DejaVu Serif'], 'font.size': 9,
                     'axes.formatter.use_mathtext': True})
rows = [json.loads(l) for l in open('compare_fem.log') if l.startswith('{')]
geos = [g for g in ['spheroid', 'dumbbell'] if any(r['geo'] == g for r in rows)]
mk = {'spheroid': 's', 'dumbbell': '^'}
col = {'spline': {2: '#4c72b0', 4: '#55a868', 6: '#8172b2'}, 'fem': {1: '#c44e52', 2: '#dd8452', 3: '#937860'}}
ls = {'spline': '-', 'fem': '--'}
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.7))
for ax, key, ylabel in [(axs[0], 'e0', r'$L^2(M)$ error'), (axs[1], 'e1', r'$H^1(M)$ seminorm error')]:
    for meth, degs in [('spline', [2, 4, 6]), ('fem', [1, 2, 3])]:
        for d in degs:
            for g in geos:
                R = sorted([r for r in rows if r['method'] == meth and r['geo'] == g and r['deg'] == d], key=lambda r: r['dof'])
                if not R: continue
                ax.loglog([r['dof'] for r in R], [r[key] for r in R], ls[meth], color=col[meth][d],
                          marker=mk[g], ms=4, mfc='white', lw=1.1)
    ax.set_xlabel('degrees of freedom'); ax.set_ylabel(ylabel)
    ax.grid(True, which='major', lw=0.3, alpha=0.5)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=col['spline'][d], ls='-', lw=1.1, label=r'spline $d=%d$' % d) for d in [2, 4, 6]]
h += [Line2D([], [], color=col['fem'][k], ls='--', lw=1.1, label=r'FEM $P_%d$' % k) for k in [1, 2, 3]]
h += [Line2D([], [], color='0.3', marker=mk[g], ls='none', mfc='white', ms=4, label=g) for g in geos]
fig.legend(handles=h, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.86))
for out in ['fig_compare.pdf']:
    fig.savefig(out)

# table: for each geometry, the finest level of each method/degree
def fmt(x):
    m, e = ('%.2e' % x).split('e'); return r'$%s\,10^{%d}$' % (m, int(e))
lines = [r'\begin{table}[htbp]', r'\centering',
         r'\caption{Conformal spherical splines against parametric surface finite elements '
         r'(Dziuk $P_1$, isoparametric $P_2$, $P_3$, nodes on the exact surface) for Poisson Test 1 on the '
         r'spheroid and the dumbbell: finest level run for each degree, errors in $L^2(M)$ and the $H^1(M)$ '
         r'seminorm with, in parentheses, the convergence rate over the last refinement, and wall-clock seconds. For the splines, ``setup'' is the geometry-independent work '
         r'(smoothness matrix, null-space matrix, round-sphere stiffness), done once per triangulation '
         r'and degree and reused for every surface; ``surface'' is the per-surface work (weight at the '
         r'quadrature points, load and constraint vectors, bordered solve). For the FEM, ``setup'' is the '
         r'isoparametric mesh and ``surface'' is assembly plus solve. Same machine, single thread, '
         r'quadrature of order $12$ (splines) and $6$ (FEM).}',
         r'\label{tab:compare}', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
         r'\begin{tabular}{llrllrr}', r'\toprule',
         r'geometry & method & dof & $L^2(M)$ & $H^1(M)$ & setup (s) & surface (s) \\', r'\midrule']
first = True
def rate(R, key):
    if len(R) < 2: return None
    r1, r0 = R[-1], R[-2]
    import math
    return math.log(r0[key] / r1[key]) / math.log(r0['h'] / r1['h'])
for g in geos:
    if not first: lines.append(r'\midrule')
    first = False
    gl = g
    for meth, degs, lab in [('spline', [2, 4, 6], lambda d, r: r'spline $(%d,%d)$' % (d, r)),
                            ('fem', [1, 2, 3], lambda d, r: r'FEM $P_%d$' % d)]:
        for d in degs:
            R = sorted([r for r in rows if r['method'] == meth and r['geo'] == g and r['deg'] == d], key=lambda r: r['dof'])
            if not R: continue
            r = R[-1]
            p0, p1 = rate(R, 'e0'), rate(R, 'e1')
            f0 = fmt(r['e0']) + (' (%.2f)' % p0 if p0 else '')
            f1 = fmt(r['e1']) + (' (%.2f)' % p1 if p1 else '')
            lines.append(r'%s & %s & %d & %s & %s & %.1f & %.1f \\' % (gl, lab(d, r['r']), r['dof'], f0, f1, r['t_setup'], r['t_surf']))
            gl = ''
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
for out in ['tab_compare.tex']:
    open(out, 'w').write('\n'.join(lines) + '\n')
print('rows', len(rows), 'geos', geos)
