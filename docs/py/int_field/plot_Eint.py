from matplotlib import pyplot as plt
from matplotlib import colors
from numpy import loadtxt, linspace#, savetxt, c_
import matplotlib.ticker as tkr


''' https://matplotlib.org/stable/users/explain/colors/colormaps.html '''


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(linspace(minval, maxval, n)))
    return new_cmap

SAVEFIG = bool(1)

file = 'IntField_example.dat' # 'IntField-Y_example'
side = 4.2
markersize = 2.5
saveas = 'IntFields_water_droplet'
FS = 12
cmap = truncate_colormap(plt.get_cmap('gnuplot'), minval=0.04, maxval=0.99, n=100)

# x y z |E|^2 Ex.r Ex.i Ey.r Ey.i Ez.r Ez.i
Y, X, Z, E2 = loadtxt(file,skiprows=1,usecols=[0,1,2,3],unpack=True)

### save file with minimal data
# tmpmask = (X==min(abs(X))) | (Y==min(abs(Y))) | (Z==min(abs(Z)) )
# savetxt('IntField_example.dat',c_[Y[tmpmask], X[tmpmask], Z[tmpmask], E2[tmpmask]])

### Projection 1

mask1 = Z == min(abs(Z))
Xp1,Yp1,E2p1 = X[mask1], Y[mask1], E2[mask1]

### Projection 1

mask2 = X == min(abs(X))
Yp2,Zp2,E2p2 = Y[mask2], Z[mask2], E2[mask2]



txt = "Square modulus of the internal field in a water droplet 8 um in diameter\n"
txt += "for two projections. Incident beam propagates along the $z$ axis."


''' ────────────── PLOT alpha ────────────── '''

fig, axs = plt.subplots(1,2,figsize=(11,4.8), dpi=600)

ax=axs[0]

ax.set_title('$z=0$', fontsize=FS, pad=12)


pcm = ax.scatter(Xp1,Yp1,c=E2p1,s=markersize,marker='s', cmap=cmap,edgecolors=None)

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


pcm = ax.scatter(Yp2,Zp2,c=E2p2,s=markersize,marker='s', cmap=cmap,edgecolors=None)
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
