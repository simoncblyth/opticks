#!/usr/bin/env python
"""
QSimTest_hemisphere_polarized.py
=================================

As everything is happening at the origin it would be 
impossible to visualize everything on top of each other. 
So represent the incoming hemisphere of photons with 
points and vectors on the unit hemi-sphere with mom
directions all pointing at the origin and polarization 
vector either within the plane of incidence (P-polarized)
or perpendicular to the plane of incidence (S-polarized). 



                   .
            .              .            
        .                     .
  
     .                          .

    -------------0---------------



Notice with S-polarized that the polarization vectors Z-component is zero 


"""
import os, numpy as np
from opticks.ana.fold import Fold
from opticks.qudarap.tests.pvplt import *
import pyvista as pv

FOLD = os.path.expandvars("/tmp/$USER/opticks/QSimTest/$TEST")
GUI = not "NOGUI" in os.environ
TEST = os.environ["TEST"]


if __name__ == '__main__':
    t = Fold.Load(FOLD)

    p = t.p
    prd = t.prd
    lim = slice(0,1000)

    print( " FOLD : %s " % FOLD)
    print( "p.shape %s " % str(p.shape) )
    print( "prd.shape %s " % str(prd.shape) )
    print(" using lim for plotting %s " % lim )

    mom = p[:,1,:3]   # hemisphere of photons all directed at origin 
    pol = p[:,2,:3]   # S or P polarized 
    pos = -mom          # illustrative choice of position on unit hemisphere 

    normal = prd[:,0,:3]  # saved from qprd 
    point =  prd[:,2,:3]  # not really position but its all zeros... so will do 

    print("mom\n", mom) 
    print("pol\n", pol) 
    print("pos\n", pos) 

    pl = pvplt_plotter(label="pvplt_polarized")   

    pvplt_polarized( pl, pos[lim], mom[lim], pol[lim] )
    pvplt_lines(     pl, pos[lim], mom[lim] )


    pvplt_arrows( pl, point, normal )

    cp = pl.show() if GUI else None

   
