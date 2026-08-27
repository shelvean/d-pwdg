"""Exact-circular-boundary DtN-PWDG experiments.

Author
------
Shelvean Kapita
Department of Mathematics, Texas A&M University

Date
----
August 2026

This module extends the direction-adaptive PWDG reference implementation to
an annular computational domain with an exact circular outer boundary.  The
Dirichlet-to-Neumann map is evaluated by a truncated Fourier series, following
the circular DtN construction used in Kapita's dissertation and in
Kapita--Monk (JCAM 327, 2018).

The mesh used here consists of curvilinear annular sectors.  This is deliberate:
no polygonal approximation of either circular boundary is introduced.  Since
the local basis is Trefftz, only boundary/skeleton integrals enter the PWDG
matrix, and the polar-sector representation gives those integrals directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.special import hankel2, h2vp, jv

from .quadrature import gauss01

__author__ = "Shelvean Kapita"
__date__ = "August 2026"


@dataclass(frozen=True)
class PolarElement:
    r0: float
    r1: float
    th0: float
    th1: float


@dataclass(frozen=True)
class PolarEdge:
    kind: str              # "arc" or "radial"
    e0: int
    e1: int                # -1 on boundary
    fixed: float           # radius for arc, theta for radial
    s0: float              # theta0 for arc, r0 for radial
    s1: float              # theta1 for arc, r1 for radial
    boundary: str | None   # "inner", "outer", or None


class PolarAnnulusMesh:
    """Curvilinear polar-sector mesh of ``a < r < R``.

    Elements are exact annular sectors.  Interior circular interfaces are also
    represented as exact arcs, while radial interfaces are straight.  This
    avoids geometric error at ``r=a`` and ``r=R`` and keeps area integration
    simple in polar coordinates.
    """

    def __init__(self, a=0.5, R=1.0, nr=2, ntheta=12):
        self.a = float(a)
        self.R = float(R)
        self.nr = int(nr)
        self.ntheta = int(ntheta)
        self.rs = np.linspace(self.a, self.R, self.nr + 1)
        self.ths = np.linspace(0.0, 2.0*np.pi, self.ntheta + 1)
        self.elements: list[PolarElement] = []
        for ir in range(self.nr):
            for j in range(self.ntheta):
                self.elements.append(PolarElement(
                    self.rs[ir], self.rs[ir+1], self.ths[j], self.ths[j+1]
                ))
        self.ne = len(self.elements)
        self.centers = np.array([self._center(e) for e in self.elements])
        self.edges = self._build_edges()

    def eid(self, ir, j):
        return ir*self.ntheta + (j % self.ntheta)

    @staticmethod
    def _center(el:
        PolarElement):
        # Area-weighted radial midpoint is unnecessary as a basis origin; the
        # polar midpoint is stable and lies strictly inside the sector.
        r = 0.5*(el.r0 + el.r1)
        th = 0.5*(el.th0 + el.th1)
        return np.array([r*np.cos(th), r*np.sin(th)])

    def _build_edges(self):
        ed: list[PolarEdge] = []
        # Circular arcs.  At an interior radial level, e0 is the inner element
        # and its outward normal is +e_r; e1 is the outer element.
        for ir_level in range(self.nr + 1):
            r = self.rs[ir_level]
            for j in range(self.ntheta):
                th0, th1 = self.ths[j], self.ths[j+1]
                if ir_level == 0:
                    e0, e1, bnd = self.eid(0, j), -1, "inner"
                elif ir_level == self.nr:
                    e0, e1, bnd = self.eid(self.nr-1, j), -1, "outer"
                else:
                    e0, e1, bnd = self.eid(ir_level-1, j), self.eid(ir_level, j), None
                ed.append(PolarEdge("arc", e0, e1, r, th0, th1, bnd))

        # Radial interfaces.  e0 is the sector immediately below theta_j and
        # e1 the sector immediately above it.  Normal from e0 is +e_theta.
        for ir in range(self.nr):
            r0, r1 = self.rs[ir], self.rs[ir+1]
            for j in range(self.ntheta):
                th = self.ths[j]
                e0 = self.eid(ir, j-1)
                e1 = self.eid(ir, j)
                ed.append(PolarEdge("radial", e0, e1, th, r0, r1, None))
        return ed

    def edge_quadrature(self, edge:
        PolarEdge, order=24):
        s, w = gauss01(order)
        if edge.kind == "arc":
            th = edge.s0 + (edge.s1-edge.s0)*s
            r = edge.fixed
            pts = np.column_stack((r*np.cos(th), r*np.sin(th)))
            er = np.column_stack((np.cos(th), np.sin(th)))
            if edge.boundary == "inner":
                n0 = -er
            else:
                n0 = er
            wds = w * r * (edge.s1-edge.s0)
            return pts, wds, n0, th
        r = edge.s0 + (edge.s1-edge.s0)*s
        th = edge.fixed
        pts = np.column_stack((r*np.cos(th), r*np.sin(th)))
        et = np.array([-np.sin(th), np.cos(th)])
        n0 = np.repeat(et[None, :], len(r), axis=0)
        wds = w * (edge.s1-edge.s0)
        theta = np.full(len(r), th)
        return pts, wds, n0, theta

    def element_quadrature(self, e, order=10):
        """Tensor Gauss quadrature on one exact polar sector."""
        el = self.elements[e]
        sr, wr = gauss01(order)
        st, wt = gauss01(order)
        rr = el.r0 + (el.r1-el.r0)*sr
        tt = el.th0 + (el.th1-el.th0)*st
        Rg, Tg = np.meshgrid(rr, tt, indexing="ij")
        pts = np.column_stack((Rg.ravel()*np.cos(Tg.ravel()),
                               Rg.ravel()*np.sin(Tg.ravel())))
        W = (wr[:,None]*wt[None,:] * (el.r1-el.r0)*(el.th1-el.th0) * Rg).ravel()
        return pts, W


def dtn_eigenvalues(k, R, N):
    """Return modes and circular DtN eigenvalues for the ``exp(+i wt)`` convention.

    Outgoing waves are Hankel functions of the second kind, so

        lambda_m = k H_m^(2)'(kR) / H_m^(2)(kR).
    """
    m = np.arange(-int(N), int(N)+1)
    z = k*R
    lam = k * h2vp(m, z, 1) / hankel2(m, z)
    return m, lam


class Hankel2Field:
    """Outgoing cylindrical wave with its source inside the excluded disk."""
    def __init__(self, k, source=(0.0, 0.0)):
        self.k = float(k)
        self.source = np.asarray(source, dtype=float)

    def __call__(self, x, y):
        rr = np.hypot(np.asarray(x)-self.source[0], np.asarray(y)-self.source[1])
        return hankel2(0, self.k*rr)

    def gradient(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        dx = x-self.source[0]
        dy = y-self.source[1]
        rr = np.hypot(dx, dy)
        # d/dr H_0^(2)(kr) = -k H_1^(2)(kr)
        fac = -self.k*hankel2(1, self.k*rr)/rr
        return np.column_stack((fac*dx, fac*dy))

    def wavevector_angle(self, x, y):
        """Leading-order local wave-vector angle for the outgoing Hankel phase.

        With exp(+i wt), H_0^(2)(kr) ~ exp(-ikr), so the wave vector points
        toward the source while physical energy propagates outward.
        """
        dx = np.asarray(x)-self.source[0]
        dy = np.asarray(y)-self.source[1]
        return np.mod(np.arctan2(-dy, -dx), 2*np.pi)


class SoundSoftDiskScattering:
    """Exact scattered field for plane-wave scattering by a sound-soft disk.

    The incident field is ``exp(i k x)``.  The scattered field uses Hankel-2
    radiation, consistent with the manuscript's exp(+i omega t) convention.
    """
    def __init__(self, k, a=0.5, series_N=100):
        self.k = float(k)
        self.a = float(a)
        self.series_N = int(series_N)

    def incident(self, x, y):
        return np.exp(1j*self.k*np.asarray(x))

    def __call__(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        r = np.hypot(x,y)
        th = np.arctan2(y,x)
        out = -(jv(0,self.k*self.a)/hankel2(0,self.k*self.a))*hankel2(0,self.k*r)
        for m in range(1, self.series_N+1):
            coef = -2.0*(1j**m)*jv(m,self.k*self.a)/hankel2(m,self.k*self.a)
            out = out + coef*hankel2(m,self.k*r)*np.cos(m*th)
        return out

    def dirichlet_data(self, x, y):
        return -self.incident(x,y)


class DtNPWDG:
    """PWDG on an exact circular annulus with a truncated DtN boundary condition."""

    def __init__(self, mesh:
        PolarAnnulusMesh, k, angles, exact,
                 p_per_element=1, alpha=0.5, beta=0.5, delta=0.5,
                 dtn_N=None, edge_order=24):
        self.mesh = mesh
        self.k = float(k)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.delta = float(delta)
        self.exact = exact
        self.edge_order = int(edge_order)
        self.p = int(p_per_element)
        ang = np.asarray(angles, dtype=float)
        if ang.size != mesh.ne*self.p:
            raise ValueError("angles must contain p_per_element values per element")
        self.angles = ang.reshape(mesh.ne, self.p)
        self.q = self.k*np.stack((np.cos(self.angles), np.sin(self.angles)), axis=-1)
        self.off = np.arange(mesh.ne+1)*self.p
        self.ndof = mesh.ne*self.p
        self.dtn_N = int(np.ceil(1.5*self.k*mesh.R)+4) if dtn_N is None else int(dtn_N)
        self.A = None
        self.F = None
        self.u = None

    def sl(self,e):
        return slice(self.off[e],self.off[e+1])

    def basis(self,e,pts):
        return np.exp(1j*(pts-self.mesh.centers[e])@self.q[e].T)

    def grad_basis(self,e,pts):
        B = self.basis(e,pts)
        return 1j*B[:,:,None]*self.q[e][None,:,:]

    def uh(self,e,pts):
        return self.basis(e,pts)@self.u[self.sl(e)]

    def grad_uh(self,e,pts):
        gb = self.grad_basis(e,pts)
        return np.einsum('qlc,l->qc', gb, self.u[self.sl(e)])

    @staticmethod
    def _push(rows, cols, vals, off, test, trial, block):
        rt=np.arange(off[test],off[test+1])
        ct=np.arange(off[trial],off[trial+1])
        rr=np.repeat(rt,len(ct))
        cc=np.tile(ct,len(rt))
        rows.append(rr)
        cols.append(cc)
        vals.append(np.asarray(block).ravel())

    def _outer_trace_matrices(self):
        """Boundary sample matrices P, D and quadrature data on exact r=R."""
        pts_all=[]
        w_all=[]
        th_all=[]
        elem_all=[]
        for edge in self.mesh.edges:
            if edge.boundary != "outer":
                continue
            pts,w,n,th = self.mesh.edge_quadrature(edge,self.edge_order)
            pts_all.append(pts)
            w_all.append(w)
            th_all.append(th)
            elem_all.extend([edge.e0]*len(w))
        pts=np.vstack(pts_all)
        w=np.concatenate(w_all)
        th=np.concatenate(th_all)
        P=np.zeros((len(w),self.ndof),dtype=complex)
        D=np.zeros_like(P)
        cursor=0
        for edge in self.mesh.edges:
            if edge.boundary != "outer":
                continue
            pts_e,w_e,n_e,th_e=self.mesh.edge_quadrature(edge,self.edge_order)
            nq=len(w_e)
            e=edge.e0
            B=self.basis(e,pts_e)
            qn=np.einsum('qc,lc->ql',n_e,self.q[e])
            P[cursor:cursor+nq,self.sl(e)] = B
            D[cursor:cursor+nq,self.sl(e)] = 1j*qn*B
            cursor += nq
        return pts,w,th,P,D

    def _dtn_apply_basis(self, th, wds, P):
        m,lam=dtn_eigenvalues(self.k,self.mesh.R,self.dtn_N)
        # v_m=(1/(2 pi R)) int_Gamma v exp(-im theta) ds
        Em=np.exp(-1j*np.outer(th,m))
        C=(Em.T @ (wds[:,None]*P))/(2*np.pi*self.mesh.R)
        Eplus=np.exp(1j*np.outer(th,m))
        SP=Eplus @ (lam[:,None]*C)
        return SP

    def assemble(self):
        rows=[]
        cols=[]
        vals=[]
        F=np.zeros(self.ndof,dtype=complex)
        k=self.k
        a=self.alpha
        b=self.beta
        for edge in self.mesh.edges:
            if edge.boundary == "outer":
                continue
            pts,w,n0,th=self.mesh.edge_quadrature(edge,self.edge_order)
            e0,e1=edge.e0,edge.e1
            if e1 >= 0:
                # interior skeleton edge
                for eA,nA in ((e0,n0),(e1,-n0)):
                    for eB,nB in ((e0,n0),(e1,-n0)):
                        BA=self.basis(eA,pts)
                        BB=self.basis(eB,pts)
                        qA_nA=np.einsum('qc,lc->ql',nA,self.q[eA])
                        qA_nB=np.einsum('qc,lc->ql',nB,self.q[eA])
                        cqB_nB=np.einsum('qc,lc->ql',nB,np.conj(self.q[eB]))
                        ndot=np.einsum('qc,qc->q',nA,nB)
                        C=(-0.5j*cqB_nB[:,:,None]
                           -0.5j*qA_nB[:,None,:]
                           +1j*(b/k)*qA_nA[:,None,:]*cqB_nB[:,:,None]
                           +1j*a*k*ndot[:,None,None])
                        # C is q,test,trial after reshaping terms above carefully.
                        # Rebuild explicitly to avoid hidden broadcasting mistakes.
                        nt=self.p
                        nr=self.p
                        block=np.zeros((nt,nr),dtype=complex)
                        for it in range(nt):
                            for jr in range(nr):
                                # JCAM skeleton representation: [[grad u]]{v}
                                # - [[u]].{grad v} + penalty terms.  This is
                                # the representation paired with the DtN fluxes.
                                cqB_nA=np.einsum('qc,c->q',nA,np.conj(self.q[eB,it]))
                                coeff=(0.5j*qA_nA[:,jr]
                                       +0.5j*cqB_nA
                                       +1j*(b/k)*qA_nA[:,jr]*cqB_nB[:,it]
                                       +1j*a*k*ndot)
                                block[it,jr]=np.sum(w*BA[:,jr]*np.conj(BB[:,it])*coeff)
                        self._push(rows,cols,vals,self.off,eB,eA,block)
            else:
                # exact inner circular Dirichlet boundary
                e=e0
                B=self.basis(e,pts)
                qn=np.einsum('qc,lc->ql',n0,self.q[e])
                block=np.zeros((self.p,self.p),dtype=complex)
                for it in range(self.p):
                    for jr in range(self.p):
                        cqn=np.einsum('qc,c->q',n0,np.conj(self.q[e,it]))
                        coeff=1j*cqn + 1j*a*k
                        block[it,jr]=np.sum(w*B[:,jr]*np.conj(B[:,it])*coeff)
                self._push(rows,cols,vals,self.off,e,e,block)
                if hasattr(self.exact,'dirichlet_data'):
                    g=self.exact.dirichlet_data(pts[:,0],pts[:,1])
                else:
                    g=self.exact(pts[:,0],pts[:,1])
                for it in range(self.p):
                    # i alpha k \bar v - d_n \bar v = i(alpha k + conj(q).n) \bar v
                    cqn=np.einsum('qc,c->q',n0,np.conj(self.q[e,it]))
                    F[self.off[e]+it] += np.sum(w*g*np.conj(B[:,it])*1j*(a*k+cqn))

        if rows:
            A=sp.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),
                            shape=(self.ndof,self.ndof)).toarray()
        else:
            A=np.zeros((self.ndof,self.ndof),dtype=complex)

        # Global DtN block on the exact circular outer boundary.
        pts,w,th,P,D=self._outer_trace_matrices()
        SP=self._dtn_apply_basis(th,w,P)
        R=D-SP
        A += P.conj().T @ (w[:,None]*R)
        A += 1j*(self.delta/k) * (R.conj().T @ (w[:,None]*R))

        self.A=sp.csr_matrix(A)
        self.F=F
        return self.A,self.F

    def solve(self):
        if self.A is None:
            self.assemble()
        self.u=la.solve(self.A.toarray(),self.F,assume_a='gen')
        return self.u

    def dtn_residual_samples(self):
        pts,w,th,P,D=self._outer_trace_matrices()
        SP=self._dtn_apply_basis(th,w,P)
        return pts,w,th,(D-SP)@self.u

    def physical_residual_vector(self, dtn_weight=None):
        """Return real vector whose squared norm is the positive skeleton residual."""
        if self.u is None:
            self.solve()
        if dtn_weight is None:
            dtn_weight=self.delta
        z=[]
        k=self.k
        for edge in self.mesh.edges:
            pts,w,n0,th=self.mesh.edge_quadrature(edge,self.edge_order)
            if edge.e1 >= 0:
                u0=self.uh(edge.e0,pts)
                u1=self.uh(edge.e1,pts)
                g0=self.grad_uh(edge.e0,pts)
                g1=self.grad_uh(edge.e1,pts)
                jv=u0-u1
                jf=np.einsum('qc,qc->q',g0-g1,n0)
                z.append(np.sqrt(self.alpha*k*w)*jv)
                z.append(np.sqrt(self.beta/k*w)*jf)
            elif edge.boundary == "inner":
                u0=self.uh(edge.e0,pts)
                if hasattr(self.exact,'dirichlet_data'):
                    g=self.exact.dirichlet_data(pts[:,0],pts[:,1])
                else:
                    g=self.exact(pts[:,0],pts[:,1])
                z.append(np.sqrt(self.alpha*k*w)*(u0-g))
        pts,w,th,r=self.dtn_residual_samples()
        z.append(np.sqrt(float(dtn_weight)/k*w)*r)
        zz=np.concatenate(z)
        return np.concatenate((zz.real,zz.imag))

    def relative_l2_error(self, order=10):
        num=0.0
        den=0.0
        for e in range(self.mesh.ne):
            pts,w=self.mesh.element_quadrature(e,order)
            ue=self.exact(pts[:,0],pts[:,1])
            uh=self.uh(e,pts)
            num += np.sum(w*np.abs(uh-ue)**2)
            den += np.sum(w*np.abs(ue)**2)
        return float(np.sqrt(num/den))

    def graph_conditioning(self):
        A=self.A.toarray()
        G=(A-A.conj().T)/(2j)
        G=(G+G.conj().T)/2
        H=(A+A.conj().T)/2
        H=(H+H.conj().T)/2
        eg=la.eigvalsh(G)
        out={'cond_raw':float(np.linalg.cond(A)), 'gmin':float(eg.min()), 'gmax':float(eg.max())}
        if eg.min() > 1e-12*max(1.0,eg.max()):
            lam=la.eigvalsh(H,G)
            sig=np.sqrt(1+lam*lam)
            out['cond_gr']=float(sig.max()/sig.min())
        else:
            out['cond_gr']=np.nan
        return out


def angle_error_deg(a,b):
    d=np.angle(np.exp(1j*(np.asarray(a)-np.asarray(b))))
    return np.abs(np.rad2deg(d))
