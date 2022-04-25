#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.pvplt import *
from opticks.ana.p import * 
from opticks.ana.r import * 


if __name__ == '__main__':

    t = Fold.Load()
    PIDX = int(os.environ.get("PIDX","-1"))

    p = t.p
    r = t.r
    prd = t.prd
    h = t.h

    hitmask = np.uint32(t.h_meta.find("hitmask:"))  
    pyhit = hit__(p, hitmask)  # hits selected in python 
    assert np.all( pyhit == h ) 


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
        print("r[i,:,:3]")
        print(r[i,:,:3]) 
        print("\n\nbflagdesc_(r[i,j])")
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



    
