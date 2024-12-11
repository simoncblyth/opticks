#!/usr/bin/env python
"""

::

   ~/o/sysrap/tests/stmm_vs_sboundary_test.sh


"""

import os, numpy as np
from opticks.ana.fold import Fold
MODE = int(os.environ.get("MODE",0))

if MODE in [2,3]:
    from opticks.ana.pvplt import * 
pass


if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    frac_twopi = t.cf[:,0,0]   
    TransCoeff = t.cf[:,1,0]  
    E1_perp    = t.cf[:,2,0]


    if MODE == 3:
        fig,ax = mpplt_plotter(label="stmm_vs_sboundary_test.py")

        ax.plot( frac_twopi*np.pi*2, TransCoeff, label="TransCoeff" ) 
        #ax.set_ylim(0,1)

        ax.plot( frac_twopi*np.pi*2, E1_perp, label="E1_perp" ) 

        ax.legend()
        fig.show(); 
    pass
pass

