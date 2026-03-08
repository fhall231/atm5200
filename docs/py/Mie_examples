from numpy import cos, pi, linspace
from _litemielib import mie_S
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx



''' toward larger droplets  '''''''''''''''''''''''''''''''''''''''''''''''''''

def calc_Mie_P(costheta_, m, x):

    s1_t, s2_t = mie_S(m,x,costheta_)
    P_ = (s1_t.real**2 + s1_t.imag**2 + s2_t.real**2 + s2_t.imag**2) * 0.5
    return P_           ### l'integrale deve fare ssa

def calc_P_Rayleigh(x,m,costheta_):
    
    p_ = 0.125*3/pi*(1+costheta_*costheta_)*0.5
    return p_

def calc_Csca_Rayleigh(m,x,k):
    
    k3alpha = x*x*x*(m*m-1)/(m*m+2)
    Csca = 8*pi/3*k3alpha/(k*k)
    return Csca


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

### constants
m = 4/3
r = 0.05 ###[μm]
wl_ = linspace(0.4, 0.7, 8)


th_ = linspace(0,pi,371)
costh_ = cos(th_)

fig, ax = plt.subplots(figsize=(7, 3))


cNorm  = colors.Normalize(vmin=-0.3, vmax=len(wl_)-1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap('rainbow'))

lines = []
for i,wl in enumerate(wl_):
    
    x = 2*pi*r/wl
    # Qext, Qsca, Qback, g = mie(m,x)
    
    y_ = calc_Mie_P(costh_, m, x)
    colorVal = scalarMap.to_rgba(i)
    line, = ax.plot(th_/pi, y_, color=colorVal, label=fr'$x={x:.2f}$ ($\lambda={wl:.2f}$)')
    lines.append(line)
    

line0, = ax.plot(th_/pi, calc_P_Rayleigh(x,m,costh_), '--k', label='$x\ll1$ (Rayleigh)')

C_R = calc_Csca_Rayleigh(m,x,0.02)

### settings

handles,labels = ax.get_legend_handles_labels()
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xlabel(r"scattering angle $\theta\: [\pi]$ ")
# plt.ylabel(r"$\frac{\text{d}\sigma_\text{sca}}{\text{d}\Omega}$", rotation=0, fontsize=12)
plt.ylabel(r"$p(\theta)$", rotation=90, fontsize=12)
# ax.set_yscale('log')
# ax.set_ylim(0.05,.3)
plt.title(f"Phase functions of near-wavelength sphere $r = {r:.2f}$ μm")
ax.grid(linewidth=0.3)
plt.show()

