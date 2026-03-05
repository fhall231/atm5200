import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numpy import linspace, pi, sqrt
from scipy.special import spherical_jn as j_sph
from scipy.special import spherical_yn as y_sph
from scipy.special import jv

def main():
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    ### init values
    
    r_ = linspace(1E-7,3,371)
    k = 2*pi
    n = 1
    x_ = k*r_
    
    j_ = j_sph(n,x_)
    y_ = y_sph(n,x_)
    J_ = jv(n,x_)
    J_hi = jv(n+0.5,x_)/sqrt(x_)
   
    ### curves

    (line_1,) = ax.plot(r_, j_, '-', lw=1.1,c='#ff9900',alpha=0.9, label='$j_n(kr)$')
    (line_2,) = ax.plot(r_, y_, '-', lw=1.1,c='#00aaff',alpha=0.9, label='$y_n(kr)$')
    (line_3,) = ax.plot(r_, J_, '--', lw=1.2,c='#ee0044',alpha=0.9, label='$J_n(kr)$')
    (line_4,) = ax.plot(r_, J_hi, '--', lw=1.2,c='#44bb00',alpha=0.9, label=r'$J_{n+1/2}(kr)/\sqrt{x}$')
    
    ### settings

    ax.set_title("Bessel functions",fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("r",fontsize=12)
    ax.set_ylabel(r"Spherical Bessel functions + $J_n$",fontsize=12)
    ax.legend(loc="upper right",prop={'size': 12},handletextpad=.5)
    ax.set_ylim(-1.1,1.1)


    ### make sliders to set n, k
    
    ax_sl1 = fig.add_axes([0.14, 0.03, 0.36, 0.03])
    ax_sl2 = fig.add_axes([0.14, 0.07, 0.36, 0.03])

    sl1 = Slider(ax_sl1, "n", 0, 14, valinit=n, color='#333333', valstep=1)
    sl2 = Slider(ax_sl2, "k", 1, 20, valinit=k, color='#440077', valstep=0.1)

    def update(dummy):

        n = sl1.val
        k = sl2.val
        x_ = k*r_
        
        line_1.set_ydata(j_sph(n,x_))
        line_2.set_ydata(y_sph(n,x_))
        line_3.set_ydata(jv(n,x_))
        line_4.set_ydata(jv(n+0.5,x_)/sqrt(x_))

        fig.canvas.draw_idle()
        
        return
        
    sl1.on_changed(update)
    sl2.on_changed(update)
    fig.subplots_adjust(bottom=0.2)
    plt.show(block=True)

if __name__ == "__main__": main()
