"""Raw Mayavi panels for the flat-quotient gallery: torus of revolution,
stereographic image of the Clifford torus, figure-eight Klein bottle,
Mobius band. Light surfaces, mesh curves of the criss-cross mesh as tubes,
no text. Compose and annotate in LaTeX."""
import numpy as np
from mayavi import mlab
mlab.options.offscreen = True
from flat_maya import render
from flat_figs import torus, mobius, klein8
def clifford(x, y):
    u, v = 2*np.pi*x, 2*np.pi*y
    p = np.array([np.cos(u), np.sin(u), np.cos(v), np.sin(v)])/np.sqrt(2)
    # stereographic projection from (0,0,0,1)
    return 2.2*p[:3]/(1 - p[3])
if __name__ == '__main__':
    render(torus, 8, 8, dict(azimuth=-60, elevation=55, distance='auto'), '/home/claude/work/hyp/paper/p_torus.png', tube=0.03)
    render(clifford, 8, 8, dict(azimuth=-60, elevation=50, distance='auto'), '/home/claude/work/hyp/paper/p_clifford.png', tube=0.03)
    render(klein8, 8, 8, dict(azimuth=-40, elevation=65, distance='auto'), '/home/claude/work/hyp/paper/p_klein.png', tube=0.02)
    render(mobius, 12, 3, dict(azimuth=-50, elevation=50, distance='auto'), '/home/claude/work/hyp/paper/p_mobius.png', tube=0.018)
