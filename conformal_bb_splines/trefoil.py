"""Tube of constant radius about a trefoil knot, parametrized by (s, theta) on
the periodic rectangle [0,2pi]x[0,2pi] with a rotation-minimizing (Bishop)
frame whose holonomy is removed. The surface is a genus-one two-manifold;
its metric in these coordinates is orthogonal, E = (1 - r kappa cos(theta -
phi0))^2, F = 0, G = r^2, and is not conformal, so the spline assembly uses the
full metric tensor: K_ab = int sqrt(g) g^{ij} d_i B_a d_j B_b, M_ab = int
sqrt(g) B_a B_b. The knotting is extrinsic: the metric depends on the
curvature function kappa(s) and the frame holonomy only, so the spectrum of
the tube is that of any tube with the same curvature profile and radius."""
import numpy as np
from scipy.interpolate import CubicSpline
RADIUS = 0.45
def core(t):
    return np.array([(2+np.cos(3*t))*np.cos(2*t), (2+np.cos(3*t))*np.sin(2*t), np.sin(3*t)])
def dcore(t, h=1e-6): return (core(t+h)-core(t-h))/(2*h)

_S = np.linspace(0, 2*np.pi, 4001)
def _frames():
    T = np.array([dcore(s) for s in _S]); sp = np.linalg.norm(T, axis=1); T = T/sp[:, None]
    N = np.zeros_like(T); v = np.array([0., 0., 1.]); n = v-(v@T[0])*T[0]; N[0] = n/np.linalg.norm(n)
    for i in range(1, len(_S)):
        n = N[i-1]-(N[i-1]@T[i])*T[i]; N[i] = n/np.linalg.norm(n)
    ang = np.arctan2(np.cross(N[0], N[-1])@T[-1], N[0]@N[-1])
    for i, s in enumerate(_S):
        a = -ang*s/(2*np.pi); b = np.cross(T[i], N[i]); N[i] = np.cos(a)*N[i]+np.sin(a)*b
    return T, N, sp
_T, _N, _SP = _frames()
_Tspl = CubicSpline(_S, _T, axis=0); _Nspl = CubicSpline(_S, _N, axis=0)
def frame(s):
    s = np.mod(s, 2*np.pi)
    t = _Tspl(s); n = _Nspl(s)
    t /= np.linalg.norm(t); n -= (n@t)*t; n /= np.linalg.norm(n); return t, n, np.cross(t, n)
# arclength reparametrization of the core, so that the parameter rectangle
# [0, L] x [0, 2 pi r] carries the induced metric with cells of comparable size
_cum = np.concatenate([[0.0], np.cumsum(0.5*(_SP[1:]+_SP[:-1])*np.diff(_S))])
L = float(_cum[-1]); CIRC = 2*np.pi*RADIUS
_tspl = CubicSpline(_cum, _S)
def t_of_s(s): return _tspl(np.mod(s, L))
def emb(s, th):
    """s in [0, L] arclength along the core, th in [0, CIRC] arclength around it"""
    t = t_of_s(s); T, n, b = frame(t); ph = th/RADIUS
    return core(t)+RADIUS*(np.cos(ph)*n+np.sin(ph)*b)
def metric(s, th, h=1e-5):
    """first fundamental form (E, F, G) by central differences of the embedding"""
    xs = (emb(s+h, th)-emb(s-h, th))/(2*h); xt = (emb(s, th+h)-emb(s, th-h))/(2*h)
    return xs@xs, xs@xt, xt@xt
