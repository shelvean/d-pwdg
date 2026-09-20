import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from cmcrameri import cm as ccm
from torusviz import *
plt.rcParams.update({'font.family':'cmr10','mathtext.fontset':'cm','axes.unicode_minus':False,'axes.formatter.use_mathtext':True})
sel=[1,7,10,14,17,20,24,27];fig,axs=plt.subplots(2,4,figsize=(14,8.6))
for ax,k in zip(axs.ravel(),sel):
    draw(ax,field(k,72),ccm.vik);ax.set_title(f'mode {k}\neigenvalue {vals[k]:.4g}',fontsize=20,pad=2,linespacing=1.15)
fig.subplots_adjust(left=.01,right=.99,top=.90,bottom=.07,wspace=.02,hspace=.22)
fig.text(.5,.02,'surfaces where the scaled eigenfunction equals 0.45 (brown) and minus 0.45 (blue)',ha='center',fontsize=19)
fig.savefig('figures/three_torus_modes.png',dpi=220);plt.close(fig)
