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
  
    if arg == 0:
        a, b = 0, 1   # horses mouth, _20
    elif arg == 1:
        a, b = 0, 2   # horses mouth, _0
    elif arg == 2:
        a, b = 0, 3   # horses mouth, _20_cudaFilterModePoint
    elif arg == 3:
        a, b = 0, 4   # horses mouth, _0_cudaFilterModePoint
    else:
        assert 0
    pass

    h = CFH()
    h.log = True 
    h.ylim = (50., 5e4 )
    h.suptitle = "Compare two 1M sample LS scintillation wavelength distributions in 1nm bins"

    c2cut = 10 
    h(wl.dom[:-1], wl.w[a], wl.w[b], [wl.l[a], wl.l[b]], c2cut=c2cut )

    xline = [wl.interp(0.05), wl.interp(0.95)]
    fig = one_cfplot(ok, h, xline=xline )

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


