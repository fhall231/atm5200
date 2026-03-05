import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from numpy import exp, log, linspace, inf, isinf

### constants

R = 8.31
rho = 1.0E6/18.    ### (mol/m³)
rmin, rmax = 0., 200. ### [nm]

### methods

def calc_sigma(T):
    return 0.118-T*1.55E-4

def calc_Fdeg(T):
    return 32. + 1.8*(T-273.15)

def DeltaG(r,Q,T,sigma):
    NA4pi = 6.022E16*12.56637   ###axscaling = 1E-7
    x = r*1.0E-9 ### nm to m
    g_s = NA4pi*sigma/(R*T)*x*x
    if Q>1.: g_v = NA4pi*rho/3*log(Q)*x**3
    else: g_v = 0.*x
    return (g_s - g_v), g_s, g_v

def r_crit(Q,T,sigma):
    if Q>1.: return 2.0E9*sigma/(R*T*rho*log(Q))
    else: return inf 

def printvalues(T,ps,sigma,Q,r_c):
    s1 = f'$T={T}$ K  ({calc_Fdeg(T):.1f} °F)'
    s2 = r'$p_\mathrm{s}=$'+f'{ps:.3g} Pa'
    s3 = fr'$\sigma={sigma:.4f}$ '+'Nm$^{-1}$'
    s4 = r'$p/p_\mathrm{s}=$'+f'{Q:.3f}'
    if Q>1.: s5 = r'$r_\mathrm{c}=$'+f'{r_c:.3g} nm'
    else: s5 = r'$r_\mathrm{c}=\infty$'#\nexists$'
    
    return s1+'\n'+s2+'\n'+s3+'\n'+s4+'\n'+s5

def calc_psat(T):    ### Tetens
    Tc = T-273.15
    if T>0.: return 610.8*exp( 17.27*Tc/(Tc+237.3) )
    else:   return 610.8*exp( 21.875*Tc/(Tc+265.5) )

def main():
    
    ### init values
    
    T = 273.15
    Q = 1.0
    # global ps
    
    ps = calc_psat(T)
    sigma = calc_sigma(T)
    
    r_ = linspace(rmin, rmax, 331)
    G_, S_, V_ = DeltaG(r_,Q,T,sigma)
    r_c = r_crit(Q,T,sigma)
    if isinf(r_c): G_c = inf
    else: G_c, _, _ = DeltaG(r_c,Q,T,sigma)

    ''' plot '''

    fig, ax = plt.subplots(figsize=(7, 7))
    
    ### settings
    
    label = r'$\beta\,\Delta G=\frac{4\pi\sigma}{k_\mathrm{B}T}r^2-\frac{4\pi\tilde{\rho}}{3}r^3\,\ln (p/p_\mathrm{s})$'
    tx = ax.text(7,0.7,printvalues(T,ps,sigma,Q,r_c),color='#003399',fontsize=11)
    
    ax.set_title("Gibbs free energy",fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("droplet radius [nm]",fontsize=12)
    ax.set_ylabel(r"$\beta\,\Delta G \cdot 10^7$",fontsize=12)
    ax.set_xlim(rmin,rmax)
    ax.set_ylim(-.5,1)

    ### curves

    (line_G,) = ax.plot(r_, G_, lw=1.5,c='#77cc00',alpha=0.7, label=label)
    (line_S,) = ax.plot(r_, S_, '--',lw=.9,c='#0044cc',alpha=0.7, label='surface term')
    (line_V,) = ax.plot(r_, V_, '--',lw=.9,c='#bb0000',alpha=0.7, label='volume term')
    (ptGmax, ) = ax.plot([r_c],[G_c],'or', ms=2)  
    
    curves = [line_G, line_S, line_V]
    colors = [l.get_color() for l in curves]
    for l in curves: l.set_visible(True)

    ### make sliders to set T, p/p* (not entangled)
    
    ax_sT = fig.add_axes([0.14, 0.03, 0.36, 0.03])
    ax_sQ = fig.add_axes([0.14, 0.07, 0.36, 0.03])

    s_T = Slider(ax_sT, "T [K]", 230, 300, valinit=T, color='k', valstep=0.1)
    s_Q = Slider(ax_sQ, "sat p/p*", 0.98, 1.18, valinit=Q, color='k', valstep=0.001)

    
    ### make checkbuttons to show/hide plots
    
    ax_b = fig.add_axes([0.8, 0.02, 0.15, 0.12])
    ax_b.axis('off')
    check = CheckButtons(ax_b,
        labels=['1 visible','2 visible','3 visible'],
        actives=[l.get_visible() for l in curves],
        label_props={'color': 'k'},
        frame_props={'edgecolor': colors},
        check_props={'facecolor': colors},
    )
    
    def callback(l):
        idx = int(l[0])-1
        ln = curves[idx]
        ln.set_visible(not ln.get_visible())
        if idx==0: ptGmax.set_visible(ln.get_visible())
        ln.figure.canvas.draw_idle()
        
    check.on_clicked(callback)

    def update(yada):

        T = s_T.val
        Q = s_Q.val

        ps = calc_psat(T) 

        sigma = calc_sigma(T)

        G_, S_, V_ = DeltaG(r_,Q,T,sigma)
        
        r_c = r_crit(Q,T,sigma)
        G_c, _, _ = DeltaG(r_c,Q,T,sigma)
        
        tx.set_text(printvalues(T,ps,sigma,Q,r_c))
        line_G.set_ydata(G_)
        line_S.set_ydata(S_)
        line_V.set_ydata(V_)
        ptGmax.set_data([r_c],[G_c])

        fig.canvas.draw_idle()
        

    s_T.on_changed(update)
    s_Q.on_changed(update)

    ax.legend(loc="lower left",prop={'size': 12},handletextpad=.5)
    plt.tight_layout(rect=[0, 0.10, 1, 1])

    plt.show(block=True)

if __name__ == "__main__": main()
