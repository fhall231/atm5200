from _litemielib import mie_Q
from numpy import exp, geomspace
import matplotlib.pyplot as plt

### change booleans below to set preferences

SAVEPNG = bool(1)
XISRHO = bool(0)
OVERAD = bool(0)   ### overplot anomalous diffraction

x_ = geomspace(.8, 10000, 501)
K = lambda w: 0.5 + exp(-w)/w + (exp(-w)-1)/w**2

fig, ax = plt.subplots(figsize=(7,4.1), dpi=300, sharex=True)

mr  = 1.5002

mi_ = [(0.001,'#c0f060'),
       (0.002, '#ffe000'),
       (0.01, '#f0c200'),
       (0.020, '#ff9900'),
       (0.050, '#ff4400'),
       (0.100, '#ff0022'),
       (0.200, '#770000'),
       (0.500, '#000000'),
       ]

for mi, c in mi_:

    m = mr - 1j*mi
    rho_ = 2*x_*(m - 1)
    label = f'{mr:#.2f}+{mi:#.0g}'

    Qext_, Qsca_, _, _, _ = mie_Q(m,x_)

    ssa_ad = 1 - K(4*x_*mi)/(2*K(1j*rho_).real)

    if XISRHO:
        ax.plot(-rho_.imag, Qsca_/Qext_, lw=0.7, c=c, label=label)
        ax.set_xlabel(r'$\mathfrak{Im}\rho$', fontsize= 11)
        if OVERAD: ax.plot(-rho_.imag, ssa_ad, '-',lw=0.3, c='k')
        ax.set_xlim(1E-2,1E4)

    else:
        ax.plot(x_, Qsca_/Qext_, lw=0.7, c=c, label=label)
        ax.set_xlabel(r'$x=kr$', fontsize= 11)
        if OVERAD: ax.plot(x_, ssa_ad, '-',lw=0.3, c='k')
        ax.set_xlim(0.1,1E4)

### plot settings

ax.legend(loc='upper right',prop={'size': 9})

ax.set_ylabel(r'$\omega_0 = 1 - Q_\text{abs}/Q_\text{ext}$', fontsize= 11)
ax.set_ylim(0.3,1)
ax.set_xscale('log')
ax.hlines(0.5,x_[1],x_[-1],'#999999',ls=':',lw=0.7)

plt.show()

### save img

if(SAVEPNG):
    fig.savefig('ssa_overview.png',format='png',bbox_inches='tight', pad_inches=0.0)
