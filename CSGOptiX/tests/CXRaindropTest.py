#!/usr/bin/env python
import numpy as np
import sys
import pyvista as pv

from opticks.ana.pvplt import *
from opticks.ana.fold import Fold
from opticks.ana.p import * 
from opticks.ana.r import * 

PIDX = int(os.environ.get("PIDX","-1"))

if __name__ == '__main__':
    t = Fold.Load()
    r = t.record if hasattr(t,'record') else None
    p = t.photon if hasattr(t,'photon') else None

    if p is None:
        print("FATAL : no photons loaded" ) 
        sys.exit(0)
    pass

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

    PLOT = not "NOPLOT" in os.environ
    if PLOT:
        pl = pvplt_plotter()
        pl.add_mesh( r_tube )
        pvplt_polarized( pl, r_pos, r_mom, r_pol, factor=60 )
        pl.add_point_labels(r_poly, "flag_label", point_size=20, font_size=36)
        pl.show()
    pass



    s = str(p[:10,:3])
    a = np.array( s.split("\n") + [""] ).reshape(-1,4)

    for i in range(len(a)):
        if not (PIDX == -1 or PIDX == i): continue
        if PIDX > -1: print("PIDX %d " % PIDX)

        if not r is None:
            print("r[i,:,:3]")
            print(r[i,:,:3])
            print("\n\nbflagdesc_(r[i,j])")
            for j in range(len(r[i])):
                print(bflagdesc_(r[i,j])  )
            pass
        pass

        print("\n")
        print("p")
        print("\n".join(a[i]))
        print(bflagdesc_(p[i]))
        print("\n")
    pass




