#!/usr/bin/env python
"""

::

    ipython -i QCtxTest.py

"""
import os, logging, numpy as np
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)

#np.set_printoptions(suppress=False, precision=8)   
np.set_printoptions(suppress=True, precision=4) 

class QCtxTest(object):
    FOLD = "/tmp/QCtxTest"
    JK = dict(sampledEnergy=(0,0),sampledWavelength=(0,1),sampledRI=(0,2),cosTheta=(0,3),sin2Theta=(1,0),BetaInverse=(1,3))  

    def __init__(self):
        self.a = np.load(os.path.join(self.FOLD, "cerenkov_photon.npy"))

    def __call__(self, name):
        j,k = self.JK.get(name)
        return self.a[:,j,k]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
     
    t = QCtxTest()
    title = "QCtxTest"

    e = t('sampledEnergy')
    w = t('sampledWavelength')
    r = t('sampledRI')
    c = t('cosTheta')
    s2 = t('sin2Theta')
    bi = t('BetaInverse')


    Pmin = 1.55
    Pmax = 15.5

    nbin = 100 
    e_dom = np.linspace(Pmin, Pmax, nbin)
    w_dom = np.linspace(1240./Pmax, 1240./Pmin/2, nbin)
    r_dom = np.linspace(1.5, 1.8, nbin)
    c_dom = np.linspace(0.83, 1., nbin)
    s2_dom = np.linspace(0, 0.3, nbin) 


    h_e = np.histogram(e, e_dom)
    h_w = np.histogram(w, w_dom)
    h_r = np.histogram(r, r_dom)
    h_c = np.histogram(c, c_dom)
    h_s2 = np.histogram(s2, s2_dom)

    nrows = 2
    ncols = 3 
    figsize = [12.8, 7.2]
    fig, axs = plt.subplots(nrows,ncols, figsize=figsize)

    fig.suptitle(title)


    ax = axs[0,0]
    ax.plot( h_e[1][:-1], h_e[0], label="e", drawstyle="steps" )
    ax.set_ylim( 0, h_e[0].max()*1.1 )
    ax.legend()

    ax = axs[0,1]
    ax.plot( h_w[1][:-1], h_w[0], label="w", drawstyle="steps" )
    ax.set_ylim( 0, h_w[0].max()*1.1 )
    ax.legend()

    ax = axs[0,2]
    ax.plot( h_r[1][:-1], h_r[0], label="ri", drawstyle="steps" )
    ax.set_ylim( 0, h_r[0].max()*1.1 )
    ax.legend()

    ax = axs[1,0]
    ax.plot( h_c[1][:-1], h_c[0], label="c", drawstyle="steps" )
    ax.set_ylim( 0, h_c[0].max()*1.1 )
    ax.legend()

    ax = axs[1,1]
    ax.plot( h_s2[1][:-1], h_s2[0], label="s2", drawstyle="steps" )
    ax.set_ylim( 0, h_s2[0].max()*1.1 )
    ax.legend()

    fig.show()


