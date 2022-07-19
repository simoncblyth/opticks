#!/usr/bin/env python
"""
recplot.py
============

Comparing photon record points between A and B in two different histories, 
under the assumption that the histories should really be the same. 
Also geometry intersect position comparison between A and B as a check of the 
translation.


"""
import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.p import *       # including cf loaded from CFBASE
from opticks.ana.eget import efloatlist_, elookce_, elook_epsilon_, eint_
from opticks.ana.pvplt import *

GEOM = os.environ.get("GEOM", None)
GDMLPath = os.environ.get("%s_GDMLPath" % GEOM, None)
GDMLSub  = os.environ.get("%s_GDMLSub"  % GEOM, None)


BOFF = efloatlist_("BOFF", "0,0,0")


def make_local(gpos_, w2m):
    gpos = np.zeros( (len(gpos_), 4 ), dtype=np.float32 )
    gpos[:,:3] = gpos_
    gpos[:,3] = 1.  
    lpos_ = np.dot( gpos, w2m )
    lpos = lpos_[:,:3] 
    return lpos


if __name__ == '__main__':
    a = Fold.Load("$A_FOLD", symbol="a")
    b = Fold.Load("$B_FOLD", symbol="b")

    if GEOM == "J000":
        if GDMLSub == None: 
            qa = "TO BT BT BT SD"
            qb = "TO BT BT BT BT BT SD"
        else:
            qa = "TO BT BT BT BT SD"       # GDMLSub wrap does not skip the hatbox
            qb = "TO BT BT BT BT BT SD"
        pass
    elif GEOM == "hama_body_log":
        qa = "TO BT SD" 
        qb = "TO BT SD" 
    else:
        assert 0, GEOM
    pass

    na = len(qa.split())
    nb = len(qb.split())
    wa = np.where(a.seq[:,0] == cseqhis_(qa))[0]
    wb = np.where(b.seq[:,0] == cseqhis_(qb))[0]
    wc = np.intersect1d(wa, wb)  # indices in common 

    print("qa : %20s : len(wa) %d " % (qa,len(wa))) 
    print("qb : %20s : len(wb) %d " % (qb,len(wb))) 
    print(" len(wc) %d " % len(wc) ) 

    # pt: point index 
    apos_ = lambda pt:a.record[wc,pt,0,:3] 
    bpos_ = lambda pt:b.record[wc,pt,0,:3] 


    assert np.all( a.sframe.m2w == b.sframe.m2w )   
    assert np.all( a.sframe.w2m == b.sframe.w2m )   

    w2m = a.sframe.w2m  


    pl = pvplt_plotter()
    pvplt_viewpoint(pl)

    for pt in range(1,na):
        gpos_ = apos_(pt)
        lpos = make_local(gpos_, w2m )
        pl.add_points( lpos, color="red" )
    pass
    for pt in range(1,nb):
        gpos_ = bpos_(pt)
        lpos = make_local(gpos_, w2m ) 
        lpos += BOFF
        pl.add_points( lpos, color="blue" )
    pass

    pl.show() 

 
