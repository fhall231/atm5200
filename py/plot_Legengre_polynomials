import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numpy import linspace, pi, cos
from scipy.special import eval_legendre



def main():
    
    ''' plot '''
    
    fig, ax = plt.subplots(figsize=(8, 6.3))
    
    ### init values
    
    n = 1
    t_ = linspace(0,pi,371)
    mu_ = cos(t_)
    
    P0_ = eval_legendre(n-1, mu_)
    P1_ = eval_legendre(n, mu_)
    P2_ = eval_legendre(n+1, mu_)
    Pf_ = mu_*P1_ - P0_


    ### curves

    (line_1,) = ax.plot(t_, P0_, '-',  lw=1.1,c='#2200ff', label=r'$P_{n-1}(\cos\theta)$')
    (line_2,) = ax.plot(t_, P1_, '-',  lw=1.1,c='#00dd66', label=r'$P_{n}(\cos\theta)$')
    (line_3,) = ax.plot(t_, P2_, '-',  lw=1.1,c='#ee0055', label=r'$P_{n+1}(\cos\theta)$')
    (line_4,) = ax.plot(t_, Pf_, '--', lw=1.0,c='#000000', label=r'$\mu\,P_{n}(\mu)-P_{n-1}(\mu)$')

    
    ax.set_title(r'$ P_{n+1}(\mu) = \frac{2n+1}{n+1}\, \mu\,P_n(\mu) - \frac{n}{n+1}\,P_{n-1}(\mu)$, where $\mu=\cos\theta$',fontsize=12, pad=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r"$\theta$",fontsize=12)
    ax.set_ylabel(r"Legendre polynomials",fontsize=12)

    ax.legend(loc="lower left",prop={'size': 11},handletextpad=.5)
    ax.set_ylim(-1.1,1.1)


    ### slider to set n
    
    ax_sl1 = fig.add_axes([0.1, 0.03, 0.36, 0.03])
    sl1 = Slider(ax_sl1, "n", 1, 20, valinit=n, color='#333333', valstep=1)

    def update(dummy):

        n = sl1.val
        
        mu_ = cos(t_)
        
        P0_ = eval_legendre(n-1, mu_)
        P1_ = eval_legendre(n, mu_)
        P2_ = eval_legendre(n+1, mu_)
        Pf_ = mu_*P1_ - P0_
        
        line_1.set_ydata(P0_)
        line_2.set_ydata(P1_)
        line_3.set_ydata(P2_)
        line_4.set_ydata(Pf_)

        fig.canvas.draw_idle()
        
        return
        
    sl1.on_changed(update)
    fig.subplots_adjust(bottom=0.14)
    plt.show(block=True)

if __name__ == "__main__": main()
