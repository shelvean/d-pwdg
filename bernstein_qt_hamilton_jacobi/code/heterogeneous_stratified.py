import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

class StratifiedEikonal:
    """Smooth heterogeneous isotropic medium with an exact single-arrival solution.

    Slowness n=n(y), phase/travel time u(x,y)=alpha*x+phi(y),
    phi'(y)=sqrt(n(y)^2-alpha^2).  Since alpha>0, x=0 is an inflow boundary.
    """
    def __init__(self, alpha=0.80):
        self.alpha=float(alpha)
        yy=np.linspace(0.0,1.0,20001)
        pp=np.sqrt(self.n_y(yy)**2-self.alpha**2)
        phi=np.concatenate([[0.0], cumulative_trapezoid(pp,yy)])
        self._phi=PchipInterpolator(yy,phi,extrapolate=True)
    def n_y(self,y):
        # high-slowness layer/lens centered at y=0.55 plus a weaker fast layer below
        y=np.asarray(y)
        return 1.08 + 0.42*np.exp(-((y-0.58)/0.16)**2) - 0.12*np.exp(-((y-0.22)/0.11)**2)
    def n(self,x): return self.n_y(np.asarray(x)[:,1])
    def n2(self,x):
        n=self.n(x);return n*n
    def phi(self,y): return self._phi(np.asarray(y))
    def u(self,x):
        x=np.asarray(x);return self.alpha*x[:,0]+self.phi(x[:,1])
    def grad(self,x):
        x=np.asarray(x);n=self.n_y(x[:,1]); py=np.sqrt(n*n-self.alpha**2)
        return np.column_stack([np.full(len(x),self.alpha),py])
    def inflow_trace(self,y): return self.phi(y)
