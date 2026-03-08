from _litemielib import mie_S_, mie_Q
from numpy import sin, cos, loadtxt, conj
from matplotlib import pyplot as plt
from scipy.integrate import simpson



def load_adda_pars(root,folder):
    
    with open(root+folder+'log','r',) as f:
        for line in f:
            
            if line.startswith('lambda'):
                wl = float( line.split(':')[1] ) 
                print('wl =', wl)

            if line.startswith('refractive index'):
                m = complex( line.split(':')[1].replace("i","j") )
                print('m =', m)

            if line.startswith('Volume-equivalent size'):
                x = float( line.split(':')[1] )
                print('x =', x)
                
            elif line.startswith('nhere we go'):
                break 

    k2 = (6.283185/wl)**2
    
    return wl, m, x, k2
    

root = 'test_runs/'
folder = 'run89550_sphere_g85_m1.8/'
# folder = 'run89551_sphere_g105_m1.5/'
# folder = 'run81828_sphere_g16_m1.55/'
# folder = 'run81473_sphere_g150_m1.203/'


''' ADDA '''

   
wl, m, x, k2 = load_adda_pars(root,folder)

tt_, S1a_r, S1a_i, S2a_r, S2a_i = loadtxt(root+folder+'ampl', usecols=(0,1,2,3,4), unpack=1, skiprows=1)

Ssq_ = 0.5 * ( S1a_r**2 + S1a_i**2 + S2a_r**2 + S2a_i**2 )
tt_ *= 0.0174533
I = 6.283185 * simpson(Ssq_*sin(tt_),tt_)


Cext, Qext, Cabs, Qabs = list(loadtxt(root+folder+'CrossSec-Y', usecols=(2)))

Cext_S1 = 12.5663/k2*S1a_r[0].real
Cext_S2 = 12.5663/k2*S2a_r[0].real

print('integrate adda |S|^2:', I)
print('k2Csca:', k2*(Cext-Cabs) )
print(f'4pi/k2 Re0 = {Cext_S1:.3f}, {Cext_S2:.3f}')
print(f'CrossSec file: {Cext:.3f}, {Qext:.3f}, {Cabs:.3f}, {Qabs:.3f}')



''' Miepython '''

S1m_, S2m_ = mie_S_(conj(m), x, cos(tt_))

Qext, Qsca, Qabs, Qbck, g = mie_Q(conj(m), x)
print(f'Miepython calc: : {Qext:.3f}, {Qsca:.3f}, {Qabs:.3f}')



''' plots '''

fig, ax = plt.subplots(figsize=(7,4),dpi=300)

ax.plot(tt_, S1a_r, '-', c='#ff2222', lw=.7, label=r'$\mathfrak{Re}\,S_1$')
ax.plot(tt_, S1m_.real, ':', c='#bb0000', lw=.7, label=r'$\mathfrak{Re}\,S_1$')

ax.plot(tt_, S1a_i, '-', c='#ff9900', lw=.7, label=r'$\mathfrak{Im}\,S_1$')
ax.plot(tt_, S1m_.imag, ':', c='#bb5500', lw=.7, label=r'$\mathfrak{Im}\,S_1$')

ax.plot(tt_, S2a_r, '-', c='#2288ee', lw=.7, label=r'$\mathfrak{Re}\,S_2$')
ax.plot(tt_, S2m_.real, ':', c='#0055aa', lw=.7, label=r'$\mathfrak{Re}\,S_2$')

ax.plot(tt_, S2a_i, '-', c='#88dd22', lw=.7, label=r'$\mathfrak{Im}\,S_2$')
ax.plot(tt_, S2m_.imag, ':', c='#33aa00', lw=.7, label=r'$\mathfrak{Im}\,S_2$')

# ax.set_yscale('log')
ax.set_xlabel(r'$\vartheta$ [rad]')
ax.set_ylabel(r'$S$ [sr$^{-\frac{1}{2}}$]')

ax.legend(loc='upper right')

