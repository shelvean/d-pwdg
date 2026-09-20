"""A smoothly curved intrinsic three-torus and exact manufactured source.

The coordinate domain is [0,1)^3 with opposite faces identified; the metric
is defined on the quotient by its periodic positive-definite tensor.  The FEM
never uses an ambient Euclidean embedding of the Riemannian manifold.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .metric import CallableMetric

TWOPI = 2.0*np.pi


def conformal_factor(x):
    """phi and grad(phi) for g=e^(2 phi) I on a flat-coordinate three-torus."""
    x = np.asarray(x, dtype=float)
    X,Y,Z = [TWOPI*x[...,i] for i in range(3)]
    sX,sY,sZ = np.sin(X), np.sin(Y), np.sin(Z)
    cX,cY,cZ = np.cos(X), np.cos(Y), np.cos(Z)
    phi = .12*sX*sY*sZ + .08*np.cos(X+Y)
    grad = np.stack([
        TWOPI*(.12*cX*sY*sZ - .08*np.sin(X+Y)),
        TWOPI*(.12*sX*cY*sZ - .08*np.sin(X+Y)),
        TWOPI*.12*sX*sY*cZ,
    ],axis=-1)
    return phi,grad


def exact_solution(x):
    x = np.asarray(x,dtype=float)
    X,Y,Z = [TWOPI*x[...,i] for i in range(3)]
    u = np.sin(X)+.35*np.cos(Y)+.2*np.sin(X+Y+Z)
    grad = TWOPI*np.stack([
        np.cos(X)+.2*np.cos(X+Y+Z),
        -.35*np.sin(Y)+.2*np.cos(X+Y+Z),
        .2*np.cos(X+Y+Z),
    ], axis=-1)
    lap = -(TWOPI**2)*(np.sin(X)+.35*np.cos(Y)+.6*np.sin(X+Y+Z))
    return u,grad,lap


def rhs(x,base_metric=None):
    """Manufactured source for g=e^(2phi) B with SPD constant matrix B."""
    phi,gphi = conformal_factor(x)
    u,gu,lap = exact_solution(x)
    if base_metric is None:
        return u - np.exp(-2*phi)*(lap+np.sum(gphi*gu,axis=-1))
    Binv=np.linalg.inv(np.asarray(base_metric,float))
    xx=np.asarray(x,float)
    X,Y,Z=[TWOPI*xx[...,i] for i in range(3)]
    # tr(B^-1 Hess u), including mixed derivatives of the coupled wave.
    ones=np.ones(3)
    lapB=-(TWOPI**2)*(
        Binv[0,0]*np.sin(X)+.35*Binv[1,1]*np.cos(Y)
        +.2*(ones@Binv@ones)*np.sin(X+Y+Z))
    return u-np.exp(-2*phi)*(lapB+np.einsum(
        '...i,ij,...j->...',gphi,Binv,gu))


def scalar_curvature(x,base_metric=None):
    """Scalar curvature for the intrinsic g=e^(2phi) B on the three-torus."""
    x=np.asarray(x,float)
    X,Y,Z=[TWOPI*x[...,i] for i in range(3)]
    phi,gphi=conformal_factor(x)
    sx,sy,sz=np.sin(X),np.sin(Y),np.sin(Z)
    cx,cy,cz=np.cos(X),np.cos(Y),np.cos(Z)
    t=.12*sx*sy*sz
    co=.08*np.cos(X+Y)
    H=np.zeros(x.shape[:-1]+(3,3))
    for j in range(3):
        H[...,j,j]=-(TWOPI**2)*t
    H[...,0,0]-=TWOPI**2*co
    H[...,1,1]-=TWOPI**2*co
    H[...,0,1]=H[...,1,0]=(TWOPI**2)*(.12*cx*cy*sz-co)
    H[...,0,2]=H[...,2,0]=(TWOPI**2)*(.12*cx*sy*cz)
    H[...,1,2]=H[...,2,1]=(TWOPI**2)*(.12*sx*cy*cz)
    B=np.eye(3) if base_metric is None else np.asarray(base_metric,float)
    Binv=np.linalg.inv(B)
    lapB=np.einsum('ij,...ji->...',Binv,H)
    gradnorm=np.einsum('...i,ij,...j->...',gphi,Binv,gphi)
    return np.exp(-2*phi)*(-4*lapB-2*gradnorm)


@dataclass(frozen=True)
class AffineTetrahedron:
    """Intrinsic chart: x = origin + E @ xhat with physical x periodic."""
    vertices: np.ndarray

    @property
    def E(self):
        V=self.vertices
        return (V[1:]-V[0]).T

    @property
    def origin(self):
        return self.vertices[0]

    def physical_coordinates(self,xhat):
        return np.asarray(xhat)@self.E.T + self.origin

    def pulled_back_metric(self,base_metric=None):
        E=self.E
        B=np.eye(3) if base_metric is None else np.asarray(base_metric,float)
        Q=E.T@B@E
        return CallableMetric(3,lambda xhat: np.exp(
            2*conformal_factor(self.physical_coordinates(xhat))[0])*Q)
