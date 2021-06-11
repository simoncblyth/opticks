#!/usr/bin/env python
"""
evtplt.py : pyvista 3d plotting photon step points within selections
========================================================================= 

::

   run evtplt.py --tag -1  


TODO: for G4 only (-ve itag) could use the deluxe buffer dx.npy for double precision step points 


"""
import numpy as np
from opticks.ana.evt import Evt
import pyvista as pv

if __name__ == '__main__':
    from opticks.ana.main import opticks_main
    ok = opticks_main(pfx="tds3ip", src="natural")

    itag = int(ok.utag)

    a = Evt(tag="%d"%itag, src=ok.src, det=ok.det, pfx=ok.pfx, args=ok)
    if hasattr(a, 'seqhis_ana'):
        expr="a.seqhis_ana.table[0:20]"
        print(expr)
        print(eval(expr))
    pass 

    if itag < 0:
        a.sel = "TO BT BT BT BT SD"  # 0x7ccccd 
        #        -6 -5 -4 -3 -2 -1
        #         Y  M  C  B  G  R
    elif itag > 0:
        a.sel = "TO BT BT BT SD"    #  0x7cccd 
        #        -5 -4 -3 -2 -1
        #         M  C  B  G  R
    else:
        assert 0 
    pass

    post = a.rpost()     # (247, 6, 4) 
    pos = post[:,:,:3]   # (247, 6, 3)

    #xyz  # octants 
    q000 = np.logical_and( np.logical_and( pos[:,-1,0] < 0, pos[:,-1,1] < 0 ), pos[:,-1,2] < 0 )  
    q001 = np.logical_and( np.logical_and( pos[:,-1,0] < 0, pos[:,-1,1] < 0 ), pos[:,-1,2] > 0 )  
    q010 = np.logical_and( np.logical_and( pos[:,-1,0] < 0, pos[:,-1,1] > 0 ), pos[:,-1,2] < 0 )  
    q011 = np.logical_and( np.logical_and( pos[:,-1,0] < 0, pos[:,-1,1] > 0 ), pos[:,-1,2] > 0 )  
    q100 = np.logical_and( np.logical_and( pos[:,-1,0] > 0, pos[:,-1,1] < 0 ), pos[:,-1,2] < 0 )  
    q101 = np.logical_and( np.logical_and( pos[:,-1,0] > 0, pos[:,-1,1] < 0 ), pos[:,-1,2] > 0 )  
    q110 = np.logical_and( np.logical_and( pos[:,-1,0] > 0, pos[:,-1,1] > 0 ), pos[:,-1,2] < 0 )  
    q111 = np.logical_and( np.logical_and( pos[:,-1,0] > 0, pos[:,-1,1] > 0 ), pos[:,-1,2] > 0 )  

    #qos = pos[q000]   # pick an octant 
    qos = pos[q001]   # pick an octant 

    colors = {}
    colors[-1] = [1,0,0]   # R
    colors[-2] = [0,1,0]   # G
    colors[-3] = [0,0,1]   # B
    colors[-4] = [0,1,1]   # C
    colors[-5] = [1,0,1]   # M
    colors[-6] = [1,1,0]   # Y

    pl = pv.Plotter()

    #     R   G  B  C 
    qq = [-1,-2,-3,-4]
    if itag < 0:
        qq += [-5]
    pass
    
    for q in qq:
        pl.add_points( qos[:,q,:], color=colors[q] )
    pass

    pl.show()




