#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.pvplt import * 

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    frac_twopi = t.cf[:,0,0]   
    TransCoeff = t.cf[:,1,0]  
    E1_perp    = t.cf[:,2,0]


    fig,ax = mpplt_plotter(label="stmm_vs_sboundary_test.py")

    ax.plot( frac_twopi*np.pi*2, TransCoeff, label="TransCoeff" ) 
    #ax.set_ylim(0,1)

    ax.plot( frac_twopi*np.pi*2, E1_perp, label="E1_perp" ) 


    ax.legend()
    fig.show(); 


pass

