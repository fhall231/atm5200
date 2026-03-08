from matplotlib import pyplot as plt
from matplotlib import colors
from numpy import loadtxt, linspace
import matplotlib.ticker as tkr

''' https://matplotlib.org/stable/users/explain/colors/colormaps.html '''

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(linspace(minval, maxval, n)))
    return new_cmap

#═════════════════════════════════════════════════════════════════════════════

# cols: x y z |E|^2 Ex.r Ex.i Ey.r Ey.i Ez.r Ez.i

file = 'IntField-Y_example'; side = 4.2; ms = 2.5; saveas = 'IntFields_water_droplet'

SAVEFIG = bool(1)
MASK = bool(1)
PLOTX = bool(0)
FS = 12

cmap = truncate_colormap(plt.get_cmap('gnuplot'), minval=0.04, maxval=0.99, n=100)


Y, X, Z, E2 = loadtxt(file,skiprows=1,usecols=[0,1,2,3],unpack=True)


if PLOTX:
    X /= 0.4764
    Y /= 0.4764
    Z /= 0.4764
    side = 5.1; ms = 1.5

txt = "Square modulus of the internal field in a water droplet 8 um in diameter\n"
txt += "for two projections. Incident beam propagates along the $z$ axis."


'''   PLOT   '''

fig, axs = plt.subplots(1,2,figsize=(11,4.8), dpi=600)

ax=axs[0]

ax.set_title('$z=0$', fontsize=FS, pad=12)

if MASK:
    mask = Z == min(abs(Z))#(-0.00248<X) & (X<0.00248)
    Xp,Yp,E2p = X[mask], Y[mask], E2[mask]
else:
    mask = (Z < 0) & (Z>-0.5)
    Xp,Yp,E2p = X[mask], Y[mask], E2[mask]


pcm = ax.scatter(Xp,Yp,c=E2p,s=ms,marker='s', cmap=cmap,edgecolors='none' )

### plot settings

ax.set_xlabel('$x$ [µm]', fontsize=FS)
ax.set_ylabel('$y$ [µm]', fontsize=FS)
ax.tick_params(axis='both', which='both', labelsize=FS, length = 4)
ax.tick_params(axis='both', which='minor', labelsize=FS)
ax.set_ylim(-side,side)
ax.set_xlim(-side,side)

### colorbar
cb = plt.colorbar(pcm, ax=ax, pad=0.022, format=tkr.FormatStrFormatter('%.1f'))
cb.ax.tick_params(labelsize=FS)
cb.set_label(label='$|E^2|$ [AU]', size=FS)

ax=axs[1]

ax.set_title('$x=0$', fontsize=FS, pad=12)


if MASK:
    mask = X == min(abs(X))
    Yp,Zp,E2p = Y[mask], Z[mask], E2[mask]
else:
    mask = X <=0.2
    Yp,Zp,E2p = Y[mask], Z[mask], E2[mask]


pcm = ax.scatter(Yp,Zp,c=E2p,s=ms,marker='s', cmap=cmap,edgecolors='none')
cb = plt.colorbar(pcm, ax=ax, pad=0.022, format=tkr.FormatStrFormatter('%.0f'))
cb.set_label(label='$|E^2|$ [AU]', size=FS)

### plot settings

ax.set_xlabel('$y$ [µm]', fontsize=FS)
ax.set_ylabel('$z$ [µm]', fontsize=FS)
ax.tick_params(axis='both', which='both', labelsize=FS, length = 4)
ax.tick_params(axis='both', which='minor', labelsize=FS)


ax.set_ylim(-side,side)
ax.set_xlim(-side,side)

fig.text(.5, -.05, txt, ha='center')
fig.tight_layout()
plt.show()
fig.savefig(saveas+'.png', format='png', transparent=False)
