#!/usr/bin/env python
"""
wavelength_cfplot.py
=======================


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

    #a, b = 1, 3     agreement between 3:localSamples and horsesMouth:1    

    a, b = 0, 1 

    h = CFH()
    h.log = True 
    h.ylim = (50., 5e4 )

    h(wl.dom[:-1], wl.w[a], wl.w[b], [wl.l[a], wl.l[b]] )

    one_cfplot(ok, h )

    plt.ion()
    plt.show()

    print(h.lhabc)   

