#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.pvplt import *
import matplotlib.pyplot as mp
SIZE = np.array([1280, 720])

MODE=int(os.environ.get("MODE","3")) 


if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

 

    a = f.SmearNormal 
    n = f.SmearNormal_names
    nj = a.shape[1]
    assert nj == len(n)


    sigma_alpha = f.SmearNormal_meta.sigma_alpha[0] if hasattr(f.SmearNormal_meta,'sigma_alpha') else None
    polish = f.SmearNormal_meta.polish[0] if hasattr(f.SmearNormal_meta,'polish') else None

    label = "QSim_MockTest.sh :"
    if not sigma_alpha is None and len(n) > 0:
        label += " white:%s sigma_alpha:%s " % ( n[0], sigma_alpha )
    pass
    if not polish is None and len(n) > 1:
        label += " red:%s polish:%s " % ( n[1],polish ) 
    pass 
    

    if MODE == 3:
        pl = pvplt_plotter(label=label)
        pvplt_viewpoint( pl )

        pos = np.array( [[0,0,0]] )
        vec = np.array( [[0,0,1]] ) 
        pvplt_lines( pl, pos, vec )

        if nj > 0:pl.add_points( a[:,0] , color="white" )
        if nj > 1:pl.add_points( a[:,1] , color="red" )

        cpos = pl.show()

    elif MODE == 2:
         
        fig, ax = mp.subplots(figsize=SIZE/100.)
        fig.suptitle(label)
        ax.set_aspect("equal")

        if nj > 0:ax.scatter( a[:,0,0], a[:,0,1], s=0.1, c="b" )
        if nj > 1:ax.scatter( a[:,1,0], a[:,1,1], s=0.1, c="r" )
        fig.show()
 
    elif MODE == 1:

        nrm = np.array( [0,0,1], dtype=np.float32 )  ## unsmeared normal is +Z direction  
        ## dot product with Z direction picks Z coordinate 
        ## so np.arccos should be the final alpha 

        alpha_angle = np.arccos(np.dot( a[:,0], nrm )) if nj > 0 else None
        polish_angle = np.arccos(np.dot( a[:,1], nrm )) if nj > 1 else None 

        bins = np.linspace(0,0.4,100)  
        h_alpha_angle = np.histogram(alpha_angle, bins=bins )[0] if not alpha_angle is None else None
        h_polish_angle = np.histogram(polish_angle, bins=bins )[0] if not polish_angle is None else None

        fig, ax = mp.subplots(figsize=SIZE/100.)
        fig.suptitle(label)

        if not h_alpha_angle  is None:ax.plot( bins[:-1], h_alpha_angle,  drawstyle="steps-post", label="h_alpha_angle" )
        if not h_polish_angle is None:ax.plot( bins[:-1], h_polish_angle, drawstyle="steps-post", label="h_polish_angle" )
        ax.legend() 

        fig.show()
    pass
pass



