from numpy import linspace, vstack
from matplotlib.colors import LinearSegmentedColormap as linsegcm
from matplotlib import colormaps as cmp
import matplotlib.pyplot as plt

''' class to define custom colormaps to use in matplotlib '''


clearmpl = bool(1)   ### clear from matplotlib when running this module as main
name1 = 'somename'   ### list names here
namelist = [name1,]


class ccm:

    ''' custom colour map class '''

    def __init__(self, name: str, palette:list[str], Ns: int=231, gamma: float=1.0, bg: str=None):
        self.name = name
        self.palette = palette    ### list of strings with hex colours, e.g. '#3399ff'
        self.Ns = Ns              ### sampling
        self.bg = bg              ### set zeroes to some colour of your choice
        self._gamma = gamma       ### gamma as in cameras, can stay 1.0 as default
        self._build()             ### set up everything and get started

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, g):
        self._gamma = g
        self._build()             ### (re)build the colormaps whenever gamma changes

    def _build(self):
        self.cm = linsegcm.from_list(self.name,self.palette, self.Ns, self.gamma)
        self.cm_r = linsegcm.from_list(self.name+'_r',self.palette[::-1], self.Ns, self.gamma)
        if self.bg:
            self.cm.set_bad(color=self.bg)
            self.cm_r.set_bad(color=self.bg)

    def __str__(self): return self.name

    __repr__ = __str__

    def save(self, ow=False):
        for name, cm in [(self.name, self.cm), (self.name+'_r', self.cm_r)]:
            try:
                cmp.get_cmap(name)
                if ow:
                    cmp.unregister(name)
                    cmp.register(cm, name)
                else:
                    print(f"Warning: '{name}' already exists. Use save(ow=True) to overwrite.")
            except ValueError:
                cmp.register(cm, name)

    def wash(self):
        try: cmp.unregister(self.name)
        except: pass
        try: cmp.unregister(self.name + '_r')
        except: pass


def trim_cm(cmap, minval=0.0, maxval=1.0, n=100):

    ''' truncate colormap if extremes are too dark '''

    new_cmap = linsegcm.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(linspace(minval, maxval, n)))
    return new_cmap


''' test '''

if __name__ == "__main__":

    palette = [
                '#0044dd', # 0.0 ### set list of colours here, see also https://htmlcolorcodes.com/
                '#911627', # 0.2
                '#e23318', # 0.4
                '#330099', # 0.6
                '#ffaa00', # 0.8
                '#f0dd00', # 1.0
                ]

    newcm = ccm('somename',palette)

    testcm = linsegcm.from_list('awexryfnd',palette, N=111)

    fig, ax = plt.subplots(figsize=(7.4, 1))

    gradient = vstack((linspace(0, 1, 256), linspace(0, 1, 256)))
    ax.imshow(gradient, aspect='auto', cmap=testcm)
    ax.set_axis_off()
    plt.show()

    if clearmpl:

        ### only clear cms in this module

        for cm in theccms:
            try: cmp.unregister(cm)
            except: pass
            try: cmp.unregister(cm)
            except: pass
