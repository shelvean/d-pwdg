from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import ELEGANT_CMAP, LINE_BLUE, LINE_RED, LINE_GOLD, LIGHT_GRID

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIG = ROOT / 'figures'

ref = np.load('/mnt/data/marmousi_reference_raw.npz')
v = ref['v']; T = ref['T']; x = ref['x']; z = ref['z']; src = ref['source']
extent = [x[0], x[-1], z[-1], z[0]]

d3 = pd.read_csv(DATA/'marmousi_raw_p3.csv')
d5 = pd.read_csv(DATA/'marmousi_raw_p5.csv')
d7 = pd.read_csv(DATA/'marmousi_raw_p7.csv')

fig, axs = plt.subplots(2, 2, figsize=(10.5, 5.8), constrained_layout=True)

im=axs[0,0].imshow(v, extent=extent, aspect='auto', cmap=ELEGANT_CMAP,
                   vmin=float(v.min()), vmax=float(v.max()))
axs[0,0].plot(src[0],src[1],marker='x',color='black',ms=5,mew=1.2)
axs[0,0].set_title('Rough Marmousi velocity')
axs[0,0].set_ylabel('Depth (km)')
fig.colorbar(im, ax=axs[0,0], label='km/s')

im=axs[0,1].imshow(T, extent=extent, aspect='auto', cmap=ELEGANT_CMAP,
                   vmin=float(T.min()), vmax=float(T.max()))
axs[0,1].plot(src[0],src[1],marker='x',color='black',ms=5,mew=1.2)
axs[0,1].set_title('Native causal first arrival')
fig.colorbar(im, ax=axs[0,1], label='Travel time (s)')

for d, col, lab, marker in [(d3,LINE_BLUE,'p=3','o'),(d5,LINE_RED,'p=5','s'),(d7,LINE_GOLD,'p=7','^')]:
    axs[1,0].plot(d.ntri,d.relrmse,marker+'-',color=col,label=lab,lw=1.8,ms=4.5)
axs[1,0].set_xlabel('Triangles'); axs[1,0].set_ylabel('Relative RMS error')
axs[1,0].grid(True,alpha=LIGHT_GRID); axs[1,0].legend(frameon=False,ncol=3)
axs[1,0].set_title('h-refinement at low order')

for d, col, lab, marker in [(d3,LINE_BLUE,'p=3','o'),(d5,LINE_RED,'p=5','s'),(d7,LINE_GOLD,'p=7','^')]:
    axs[1,1].plot(d.dofs,d.relrmse,marker+'-',color=col,label=lab,lw=1.8,ms=4.5)
axs[1,1].set_xlabel('Global C$^0$ coefficients'); axs[1,1].set_ylabel('Relative RMS error')
axs[1,1].grid(True,alpha=LIGHT_GRID); axs[1,1].set_xscale('log')
axs[1,1].set_title('Accuracy versus representation size')

fig.savefig(FIG/'marmousi_raw_lowp.pdf',bbox_inches='tight')
fig.savefig(FIG/'marmousi_raw_lowp.png',dpi=220,bbox_inches='tight')
plt.close(fig)
