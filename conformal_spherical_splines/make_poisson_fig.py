import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex': False, 'mathtext.fontset': 'cm', 'font.family': 'serif', 'font.serif': ['cmr10','DejaVu Serif'], 'font.size': 9, 'axes.formatter.use_mathtext': True})
rows=np.load('poisson_rows_ext.npy',allow_pickle=True)
geos=['sphere','spheroid','dumbbell']
mk={'sphere':'o','spheroid':'s','dumbbell':'^'}
ls={2:'-',4:'--',6:':'}
col={2:'#4c72b0',4:'#55a868',6:'#8172b2'}
fig,axs=plt.subplots(1,2,figsize=(6.3,2.6))
for ax,key,slope_off,ylabel in [(axs[0],'e0',1,r'$L^2(M)$ error'),(axs[1],'e1',0,r'$H^1(M)$ seminorm error')]:
    for d in [2,4,6]:
        for g in geos:
            R=[r for r in rows if r['geo']==g and r['d']==d]
            h=[r['h'] for r in R]; e=[r[key] for r in R]
            ax.loglog(h,e,ls[d],color=col[d],marker=mk[g],ms=4,mfc='white',lw=1.1)
        # reference slope
        R=[r for r in rows if r['geo']=='sphere' and r['d']==d]
        h=np.array([r['h'] for r in R]); e=np.array([r[key] for r in R])
        p=d+slope_off
        hh=np.array([h[-1],h[-2]]); ref=e[-1]*(hh/h[-1])**p*2.2
        ax.loglog(hh,ref,'-',color='0.35',lw=0.8)
        ax.text(hh[1]*1.05,ref[1]*1.05,r'$%d$'%p,fontsize=8,color='0.25',ha='left',va='bottom')
    hs=sorted({round(r['h'],3) for r in rows},reverse=True)
    ax.set_xticks(hs); ax.set_xticklabels([r'$%.3f$'%x for x in hs]); ax.minorticks_off()
    ax.set_xlabel(r'geodesic mesh size $h$'); ax.set_ylabel(ylabel)
    ax.grid(True,which='major',lw=0.3,alpha=0.5)
from matplotlib.lines import Line2D
hd=[Line2D([],[],color=col[d],ls=ls[d],lw=1.1,label=r'$d=%d$'%d) for d in [2,4,6]]
hg=[Line2D([],[],color='0.3',marker=mk[g],ls='none',mfc='white',ms=4,label=g) for g in geos]
fig.legend(handles=hd+hg,loc='upper center',ncol=6,frameon=False,bbox_to_anchor=(0.5,1.02),fontsize=8)
fig.tight_layout(rect=(0,0,1,0.92))
fig.savefig('fig_poisson_conv.pdf')
print('saved')
