#!/usr/bin/env python
"""
wavelength_cfplot.py
=======================

::

    an ; 

    ARG=0 ipython -i wavelength_cfplot.py
    ARG=1 ipython -i wavelength_cfplot.py
    ARG=2 ipython -i wavelength_cfplot.py
    ARG=3 ipython -i wavelength_cfplot.py

    ARG=4 ipython -i wavelength_cfplot.py
    ARG=5 ipython -i wavelength_cfplot.py
    ARG=6 ipython -i wavelength_cfplot.py
    ARG=7 ipython -i wavelength_cfplot.py
    ARG=8 ipython -i wavelength_cfplot.py
    ARG=9 ipython -i wavelength_cfplot.py
    ARG=10 ipython -i wavelength_cfplot.py
    ARG=11 ipython -i wavelength_cfplot.py


    mkdir -p ~/simoncblyth.bitbucket.io/env/presentation/ana/wavelength_cfplot
    cp /tmp/ana/wavelength_cfplot/*.png ~/simoncblyth.bitbucket.io/env/presentation/ana/wavelength_cfplot/


"""

import os, numpy as np, logging
log = logging.getLogger(__name__)

from opticks.ana.main import opticks_main 
from opticks.ana.key import keydir
from opticks.ana.material import Material
from opticks.ana.wavelength import Wavelength
from opticks.ana.cfplot import one_cfplot
from opticks.ana.cfh import CFH

from matplotlib import pyplot as plt 

if __name__ == '__main__':
    ok = opticks_main()
    kd = keydir(os.environ["OPTICKS_KEY"])
    wl = Wavelength(kd)
    arg = int(os.environ.get("ARG","0")) 
    a,b = wl.cf(arg)

    wa = wl.w[a] 
    wb = wl.w[b]

    la = wl.l[a]
    lb = wl.l[b]

    energy = False
    hc_eVnm = 1240.

    dom = wl.dom[:-1]

    if energy:
        wa = hc_eVnm/wa
        wb = hc_eVnm/wb
        dom = hc_eVnm/dom[::-1]
    pass

    h = CFH()

    if arg < 4:
        h.log = True 
        h.ylim = (50., 5e4 )
        title = "Compare two 1M sample LS scintillation wavelength distributions in 1nm bins" 
        xline = [wl.interp(0.05), wl.interp(0.95)]
    else:
        h.log = True
        title = "Compare two 2.82M samples of Cerenkov wavelength distributions in 1nm bins : poor chi2, interpolation artifact ?  " 
        xline = []
    pass 

    h.suptitle = title
    c2cut = 10 
    h(dom, wa, wb, [la, lb], c2cut=c2cut )

    fig, axs = one_cfplot(ok, h, xline=xline )

    if 1:

        rindex = np.load(os.path.join(kd, "GScintillatorLib/LS_ori/RINDEX.npy"))
        rindex[:,0] *= 1e6   
        if not energy:
            rindex[:,0] = hc_eVnm/rindex[:,0]
            rindex = rindex[::-1]       
        pass

        ax = axs[0]

        ylim = ax.get_ylim()
        for i in range(len(rindex)):
            ax.plot( [rindex[i,0], rindex[i,0]], ylim , linestyle="dotted", color="b" )
        pass

        axr = ax.twinx() 
        axr.set_ylabel("rindex")
        axr.spines['right'].set_position(('outward', 0))

        p3, = axr.plot( rindex[:,0] , rindex[:,1], drawstyle="steps-post", label="rindex", color="b" )

        #wmin, wmax = 80., 800.
        wmin, wmax = 100., 400.
        emin, emax = hc_eVnm/wmax, hc_eVnm/wmin
                
        if energy:
            xlim = [emin, emax]
        else:
            xlim = [wmin, wmax]
        pass

        axs[0].set_xlim(*xlim)
        axs[1].set_xlim(*xlim)

        axs[1].set_ylim(0,h.chi2.max()*1.1)
    pass

    plt.ion()
    plt.show()

    print(h.lhabc)   

    path="/tmp/ana/wavelength_cfplot/%s_%s.png" % ( wl.l[a], wl.l[b] )
    fold=os.path.dirname(path)
    if not os.path.isdir(fold):
       os.makedirs(fold)
    pass 
    log.info("save to %s " % path)
    fig.savefig(path)


