# Is the Evans-Fung red blood cell a radial graph over its center, and how marginally?
import numpy as np
R0=3.91; C0,C2,C4=0.207161,2.002558,-1.122762
e=np.linspace(1e-4,np.pi/2-1e-4,400001)
r=R0*np.sin(e); z=0.5*R0*np.cos(e)*(C0+C2*np.sin(e)**2+C4*np.sin(e)**4)
rho=np.hypot(r,z); psi=np.arctan2(r,z)
drho=np.gradient(rho,e); dpsi=np.gradient(psi,e)
print('psi monotone in eta (radial graph):', (dpsi>0).all())
theta=np.arctan2(rho*np.abs(dpsi),np.abs(drho))
i=np.argmin(theta)
print('min angle between ray and surface: %.2f deg at psi = %.1f deg' % (np.degrees(theta[i]),np.degrees(psi[i])))
print('max |rho_psi|/rho = %.2f' % (np.abs(drho/dpsi)/rho).max())
