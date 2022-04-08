#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.pvplt import *

from opticks.ana.hismask import HisMask   
hm = HisMask()

boundary_ = lambda p:p.view(np.uint32)[3,0] >> 16
flag__    = lambda p:p.view(np.uint32)[:,3,0] & 0xffff
flag_     = lambda p:p.view(np.uint32)[3,0] & 0xffff
identity_ = lambda p:p.view(np.uint32)[3,1]   
idx_      = lambda p:p.view(np.uint32)[3,2] & 0x7fffffff
orient_   = lambda p:p.view(np.uint32)[3,2] >> 31
flagmask_ = lambda p:p.view(np.uint32)[3,3]
flagdesc_ = lambda p:" %6d prd(%3d %3d %1d)  %3s  %10s " % ( idx_(p),  boundary_(p),identity_(p),orient_(p),  hm.label(flag_(p)),hm.label( flagmask_(p) ))

from opticks.CSG.CSGFoundry import CSGFoundry 
cf = CSGFoundry.Load()
bflagdesc_ = lambda p:"%s : %s " % ( flagdesc_(p), cf.bndnamedict[boundary_(p)] )


def make_record_cells(r):
    """
    * indices of the cells reference record points 

    In [1]: cells
    Out[1]: 
    array([[ 5,  0,  1,  2,  3,  4],
           [ 5,  5,  6,  7,  8,  9],
           [ 5, 10, 11, 12, 13, 14],
           [ 5, 15, 16, 17, 18, 19],
           [ 5, 20, 21, 22, 23, 24],
           [ 5, 25, 26, 27, 28, 29],
           [ 5, 30, 31, 32, 33, 34],
           [ 5, 35, 36, 37, 38, 39]])

    """
    assert r.ndim == 4  
    assert r.shape[2:] == (4,4)
    num_pho, max_rec = r.shape[:2]
    cells = np.zeros( (num_pho,max_rec+1), dtype=np.int ) 
    offset = 0 
    for i in range(num_pho):
        cells[i,0] = max_rec
        cells[i,1:] = np.arange( 0, max_rec ) + offset 
        offset += max_rec 
    pass
    return cells 



if __name__ == '__main__':

    t = Fold.Load()
    PIDX = int(os.environ.get("PIDX","-1"))

    p = t.p
    r = t.r
    prd = t.prd


    r_pos = r[:,:,0,:3].reshape(-1,3)   
    r_mom = r[:,:,1,:3].reshape(-1,3) 
    r_pol = r[:,:,2,:3].reshape(-1,3) 
    r_flag = flag__(r.reshape(-1,4,4))  

    r_flag_label = hm.label( r_flag )

    r_cells = make_record_cells( r ) 

    r_poly = pv.PolyData() 
    r_poly.points = r_pos
    r_poly.lines = r_cells 
    r_poly["flag_label"] = r_flag_label

    r_tube = r_poly.tube(radius=1) 

    PLOT = "PLOT" in os.environ
    if PLOT:
        pl = pvplt_plotter()
        pl.add_mesh( r_tube )
        pvplt_polarized( pl, r_pos, r_mom, r_pol, factor=60 )
        pl.add_point_labels(r_poly, "flag_label", point_size=20, font_size=36)
        pl.show() 
    pass


    s = str(p[:,:3]) 
    a = np.array( s.split("\n") + [""] ).reshape(-1,4)



    for i in range(len(a)):
        if not (PIDX == -1 or PIDX == i): continue 
        if PIDX > -1: print("PIDX %d " % PIDX) 
        print("r")
        print(r[i,:,:3]) 
        print("\n\nbflagdesc_")
        for j in range(len(r[i])):
            print(bflagdesc_(r[i,j])  ) 
        pass

        print("\n") 
        print("p")
        print("\n".join(a[i]))
        print(bflagdesc_(p[i]))
        print("\n") 

        print("\n\n") 
    pass



    
