#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold

MODE=int(os.environ.get("MODE","0")) 
print("A.MODE:%d" % MODE )

if MODE in [2,3]:
    from opticks.ana.pvplt import *
    import matplotlib.pyplot as mp
pass
print("B.MODE:%d" % MODE )

SIZE = np.array([1280, 720])

CHECK = os.environ.get("CHECK", "SmearNormal_SigmaAlpha")

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    a = getattr(f, CHECK) 
    n = getattr(f, "%s_names" % CHECK )
    m = getattr(f, "%s_meta" % CHECK ) 

    nj = a.shape[1]
    assert nj == 3 
    
    value = m.value[0] if hasattr(m,'value') else None
    valuename = m.valuename[0] if hasattr(m,'valuename') else None

    label = "QSim_MockTest.sh : CHECK %s " % CHECK 
    label += " white:%s %s:%s " % ( n[0], valuename, value )
    

    if MODE == 3:
        pl = pvplt_plotter(label=label)
        pvplt_viewpoint( pl )

        pos = np.array( [[0,0,0]] )
        vec = np.array( [[0,0,1]] ) 
        pvplt_lines( pl, pos, vec )

        pl.add_points( a , color="white" )

        cpos = pl.show()

    elif MODE == 2:
         
        fig, ax = mp.subplots(figsize=SIZE/100.)
        fig.suptitle(label)
        ax.set_aspect("equal")

        ax.scatter( a[:,0], a[:,1], s=0.1, c="b" )
        fig.show()
 
    elif MODE == 1:

        nrm = np.array( [0,0,1], dtype=np.float32 )  ## unsmeared normal is +Z direction  
        ## dot product with Z direction picks Z coordinate 
        ## so np.arccos should be the final alpha 

        angle = np.arccos(np.dot( a, nrm )) 

        bins = np.linspace(0,0.4,100)  
        h_angle = np.histogram(angle, bins=bins )[0] 

        fig, ax = mp.subplots(figsize=SIZE/100.)
        fig.suptitle(label)

        if not h_angle is None:ax.plot( bins[:-1], h_angle,  drawstyle="steps-post", label="h_angle" )
        ax.legend() 

        fig.show()
    pass
pass


