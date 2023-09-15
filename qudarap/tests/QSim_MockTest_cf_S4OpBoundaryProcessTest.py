#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
import matplotlib.pyplot as mp
SIZE = np.array([1280, 720])

MODE = int(os.environ.get("MODE", "1"))

if __name__ == '__main__':
    af = Fold.Load("$AFOLD", symbol="af")
    bf = Fold.Load("$BFOLD", symbol="bf")
    cf = Fold.Load("$CFOLD", symbol="cf")

    print(repr(af))
    print(repr(bf))
    print(repr(cf))

    label = "QSim_MockTest_cf_S4OpBoundaryProcessTest.sh"  
    a_label = af.base 
    b_label = bf.base
    c_label = cf.base

    a = af.q[:,:3]
    b = bf.q[:,:3]
    c = cf.q[:,:3]
    
    nrm = np.array( [0,0,1], dtype=np.float32 )  ## unsmeared normal is +Z direction  
    ## dot product with Z direction picks Z coordinate 
    ## so np.arccos should be the final alpha 

    a_angle = np.arccos(np.dot( a, nrm ))  
    b_angle = np.arccos(np.dot( b, nrm ))  
    c_angle = np.arccos(np.dot( c, nrm ))  

    bins = np.linspace(0,0.4,100)  
    a_angle_h= np.histogram(a_angle, bins=bins )[0] 
    b_angle_h= np.histogram(b_angle, bins=bins )[0] 
    c_angle_h= np.histogram(c_angle, bins=bins )[0] 

    if MODE == 1:
        fig, ax = mp.subplots(figsize=SIZE/100.)
        fig.suptitle(label)
        ax.plot( bins[:-1], a_angle_h,  drawstyle="steps-post", label=a_label )
        ax.plot( bins[:-1], b_angle_h,  drawstyle="steps-post", label=b_label )
        ax.plot( bins[:-1], c_angle_h,  drawstyle="steps-post", label=c_label )
        ax.legend() 
        fig.show()
    else:
        print("not impl MODE %d " % MODE )
    pass





