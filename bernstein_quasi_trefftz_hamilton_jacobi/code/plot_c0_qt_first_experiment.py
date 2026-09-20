from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import LINE_BLUE, LINE_RED, LIGHT_GRID
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data';F=ROOT/'figures'
pdta=pd.read_csv(D/'c0_qt_embedded_rotating_anisotropic_p_sweep.csv')
hdta=pd.read_csv(D/'c0_qt_embedded_rotating_anisotropic_h_sweep.csv')
fig,axs=plt.subplots(2,2,figsize=(8.2,6.0),constrained_layout=True)
ax=axs[0,0]
ax.semilogy(pdta.p,pdta.c0_relL2,'o-',color=LINE_BLUE,label=r'full $C^0$')
ax.semilogy(pdta.p,pdta.qt_relL2,'s--',color=LINE_RED,label=r'embedded $C^0$-qT')
ax.set_xticks(pdta.p);ax.set_xlabel(r'degree $p$');ax.set_ylabel(r'relative $L^2$ error');ax.grid(True,which='both',alpha=LIGHT_GRID);ax.legend(frameon=False)
ax=axs[0,1]
ax.semilogy(pdta.p,pdta.qt_selected_rms,'s-',color=LINE_RED,label='selected qT moments')
ax.semilogy(pdta.p,pdta.qt_remaining_rms,'o--',color=LINE_BLUE,label='complementary moments')
ax.set_xticks(pdta.p);ax.set_xlabel(r'degree $p$');ax.set_ylabel('moment RMS');ax.grid(True,which='both',alpha=LIGHT_GRID);ax.legend(frameon=False)
ax=axs[1,0]
ax.loglog(hdta.ntri,hdta.c0_relL2,'o-',color=LINE_BLUE,label=r'full $C^0$')
ax.loglog(hdta.ntri,hdta.qt_relL2,'s--',color=LINE_RED,label=r'embedded $C^0$-qT')
ax.set_xlabel('triangles');ax.set_ylabel(r'relative $L^2$ error');ax.grid(True,which='both',alpha=LIGHT_GRID)
ax=axs[1,1]
ax.plot(pdta.p,pdta.global_c0_dofs,'o-',color=LINE_BLUE,label=r'parent $N_h$')
ax.plot(pdta.p,pdta.global_qt_trace_dim,'s--',color=LINE_RED,label=r'qT trace $N_t$')
ax.set_xticks(pdta.p);ax.set_xlabel(r'degree $p$');ax.set_ylabel('global coordinates');ax.grid(True,alpha=LIGHT_GRID);ax.legend(frameon=False)
fig.savefig(F/'c0_qt_first_experiment.pdf',bbox_inches='tight')
fig.savefig(F/'c0_qt_first_experiment.png',dpi=220,bbox_inches='tight')
