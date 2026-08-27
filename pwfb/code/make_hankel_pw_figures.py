import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.special import hankel1
from pathlib import Path

root=Path(__file__).resolve().parents[1]
figdir=root/'figures';figdir.mkdir(exist_ok=True)
datadir=root/'data'
summary=pd.read_csv(datadir/'hankel_pw_fan_summary.csv')
arrows=pd.read_csv(datadir/'hankel_pw_fan_arrows.csv')
K=16.;a=.5;R=1.;h=.44
sources=[0.,.15,.25,.35]
labels={0.0:r'$x_s=0$',0.15:r'$x_s=0.15$',0.25:r'$x_s=0.25$',0.35:r'$x_s=0.35$'}
# shared muted colormap identical in spirit to manuscript
from matplotlib.colors import LinearSegmentedColormap
base=plt.get_cmap('RdBu_r')(np.linspace(.04,.96,256));base[:,:3]=.70*base[:,:3]+.30
CM=LinearSegmentedColormap.from_list('mutedRdBu_shared',base)
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':7.5,'xtick.labelsize':8,'ytick.labelsize':8})

# FIG 1: field + exact/recovered principal directions
xx=np.linspace(-1,1,500);X,Y=np.meshgrid(xx,xx);rr=np.hypot(X,Y);mask=(rr>=a)&(rr<=R)
fig,axs=plt.subplots(1,4,figsize=(7.3,2.15),sharex=True,sharey=True)
for ax,sx in zip(axs,sources):
    U=np.full_like(X,np.nan,dtype=float); d=np.hypot(X-sx,Y); U[mask]=np.real(hankel1(0,K*d[mask])); vm=np.nanpercentile(np.abs(U),99)
    ax.pcolormesh(X,Y,U,cmap=CM,vmin=-vm,vmax=vm,shading='auto',rasterized=True)
    t=np.linspace(0,2*np.pi,500)
    for rad in (a,R): ax.plot(rad*np.cos(t),rad*np.sin(t),'k-',lw=.65)
    for j in range(8):
        th=j*np.pi/4;ax.plot([a*np.cos(th),R*np.cos(th)],[a*np.sin(th),R*np.sin(th)],color='0.35',lw=.45)
    sub=arrows[np.isclose(arrows.source_x,sx)]
    for _,r in sub.iterrows():
        L=.12
        # exact geometric direction as thin gray segment
        ax.plot([r.cx,r.cx+L*np.cos(r.truth)],[r.cy,r.cy+L*np.sin(r.truth)],color='0.35',lw=.8,ls='--')
        # recovered principal PW direction as black arrow
        ax.annotate('',xy=(r.cx+L*np.cos(r.recovered),r.cy+L*np.sin(r.recovered)),xytext=(r.cx,r.cy),arrowprops=dict(arrowstyle='-|>',lw=.8,color='k',mutation_scale=6))
    ax.plot(sx,0,'ko',ms=2.5)
    ax.set_title(labels[sx]);ax.set_aspect('equal');ax.set_xlim(-1.03,1.03);ax.set_ylim(-1.03,1.03)
    ax.set_xticks([-1,-.5,0,.5,1]);ax.set_yticks([-1,-.5,0,.5,1]);ax.set_xlabel('$x$')
axs[0].set_ylabel('$y$')
fig.text(.5,.01,'solid arrow: recovered principal PW direction; dashed segment: geometric ray direction',ha='center',fontsize=7.5)
fig.tight_layout(rect=[0,.055,1,1],w_pad=.45)
fig.savefig(figdir/'hankel_pw_center_offcenter_fields.pdf',dpi=240,bbox_inches='tight');fig.savefig(figdir/'hankel_pw_center_offcenter_fields.png',dpi=180,bbox_inches='tight');plt.close(fig)

# FIG 2: errors vs p, one panel each source
fig,axs=plt.subplots(2,2,figsize=(6.8,5.0),sharex=True,sharey=True)
for ax,sx in zip(axs.ravel(),sources):
    s=summary[np.isclose(summary.source_x,sx)]
    for meth,ls,marker in [('ESPRIT-centered fan','-','o'),('equispaced','--','s')]:
        q=s[s.method==meth]
        ax.semilogy(q.p,q.volume_L2_error,ls,marker=marker,ms=3.5,lw=1.0,label=meth+' $L^2$')
        ax.semilogy(q.p,q.trace_error,ls,marker='x',ms=3.8,lw=.8,alpha=.65,label=meth+' trace')
    ax.set_title(labels[sx]);ax.grid(True,which='major',lw=.3,color='0.85');ax.set_xticks([1,5,9,13,17,21]);ax.set_ylim(1e-4,1.1)
    ax.set_xlabel('plane waves per sector $p$');ax.set_ylabel('relative error')
handles,lab=axs[0,0].get_legend_handles_labels();fig.legend(handles,lab,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.01),frameon=False)
fig.tight_layout(rect=[0,.0,1,.94],h_pad=1.0,w_pad=1.0)
fig.savefig(figdir/'hankel_pw_center_offcenter_convergence.pdf',dpi=240,bbox_inches='tight');fig.savefig(figdir/'hankel_pw_center_offcenter_convergence.png',dpi=180,bbox_inches='tight');plt.close(fig)

# FIG 3: conditioning and rank for ESPRIT centered fan
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6.8,2.65))
for sx in sources:
    s=summary[(np.isclose(summary.source_x,sx))&(summary.method=='ESPRIT-centered fan')]
    ax1.semilogy(s.p,s.max_gram_cond,'-o',ms=3.4,lw=.9,label=labels[sx])
    ax2.plot(s.p,s.min_rank,'-o',ms=3.4,lw=.9,label=labels[sx])
ax1.axhline(1e12,color='0.45',ls='--',lw=.7);ax1.text(1.2,1.7e12,r'$10^{12}$ rank threshold',fontsize=7,color='0.35')
ax1.set_xlabel('$p$');ax1.set_ylabel('maximum local PW Gram condition');ax1.set_xticks([1,5,9,13,17,21]);ax1.grid(True,which='major',lw=.3,color='0.85')
ax2.plot([1,21],[1,21],'k--',lw=.7,label='full rank');ax2.set_xlabel('$p$');ax2.set_ylabel('minimum retained rank');ax2.set_xticks([1,5,9,13,17,21]);ax2.set_yticks([1,5,9,13,17,21]);ax2.grid(True,lw=.3,color='0.85')
ax1.set_title('(a) trace-Gram conditioning');ax2.set_title('(b) effective PW rank')
ax1.legend(frameon=False,ncol=2,loc='upper left',fontsize=7)
fig.tight_layout(w_pad=1.3)
fig.savefig(figdir/'hankel_pw_center_offcenter_conditioning.pdf',dpi=240,bbox_inches='tight');fig.savefig(figdir/'hankel_pw_center_offcenter_conditioning.png',dpi=180,bbox_inches='tight');plt.close(fig)
print('figures written')
