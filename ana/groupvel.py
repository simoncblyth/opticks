#!/usr/bin/env python
"""

"""

from opticks.ana.xmatlib import XMatLib
from opticks.ana.PropLib import PropLib

if __name__ == '__main__':


    xma = XMatLib("/tmp/test.dae")

    mat = "MineralOil"

    g = xma[mat]["GROUPVEL"]

    n = xma[mat]["RINDEX"]




    
if 0:
    mlib = PropLib("GMaterialLib")

    mat = mlib("MineralOil")

    n = mat[:,0]
  
    w = mlib.domain

    dn = n[1:] - n[0:-1]

    dw = w[1:] - w[0:-1]



