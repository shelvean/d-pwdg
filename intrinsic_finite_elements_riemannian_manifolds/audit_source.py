import numpy as np
from riemannfem.torus_metric import conformal_factor
from oneform_high_order import analytic_oneform,B_ANISO,T

def source_by_fd(xx,B,h=8e-6):
    # Independent flux-difference construction: f=alpha + d(delta alpha) +delta(d alpha)
    C=np.linalg.inv(B);ph,_=conformal_factor(xx)
    aa,bb,ss,ff,_=analytic_oneform(xx,B)
    dsig=np.zeros_like(xx); divflux=np.zeros((len(xx),3))
    for i in range(3):
        xp=xx.copy();xp[:,i]+=h;xm=xx.copy();xm[:,i]-=h
        ap,bp,sp,fp,phip=analytic_oneform(xp,B)
        am,bm,sm,fm,phim=analytic_oneform(xm,B)
        dsig[:,i]=(sp-sm)/(2*h)
        # rho*g^-1 wedge^2 beta = e^{-phi} sqrt(detB) C^{ia} C^{jb} beta_ab
        fluxp=np.exp(-phip)[:,None]*np.einsum('a,jb,qab->qj',C[i,:],C,bp)
        fluxm=np.exp(-phim)[:,None]*np.einsum('a,jb,qab->qj',C[i,:],C,bm)
        divflux+=(fluxp-fluxm)/(2*h)
    dbeta=-np.exp(-ph)[:,None]*(divflux@B.T)
    return aa+dsig+dbeta

xx=np.array([[.17,.23,.39],[.43,.38,.77],[.61,.47,.19]])
for B in (np.eye(3),B_ANISO):
    f=analytic_oneform(xx,B)[3]
    fd=source_by_fd(xx,B)
    e=np.max(np.abs(f-fd)); print('B',B[0,0],'max analytic-versus-independent-FD source difference',e)
    assert e<2e-6
