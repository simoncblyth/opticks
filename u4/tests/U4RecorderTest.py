#!/usr/bin/env python 
"""
U4RecorderTest.py
==================

::

     18 struct spho
     19 {
     20     int gs ; // 0-based genstep index within the event
     21     int ix ; // 0-based photon index within the genstep
     22     int id ; // 0-based photon identity index within the event 
     23     int gn ; // 0-based reemission index incremented at each reemission 


"""
import numpy as np
from opticks.ana.fold import Fold
from opticks.ana.p import * 


def check_pho_labels(l):
    """ 
    :param l: spho labels 

    When reemission is enabled this would fail for pho0 (push_back labels)
    but should pass for pho (labels slotted in by event photon id)

    1. not expecting gaps in list of unique genstep index gs_u 
       as there should always be at least one photon per genstep

    2. expecting the photon identity index to be unique within event, 
       so id_c should all be 1, otherwise likely a labelling issue

    """
    gs, ix, id_, gn = l[:,0], l[:,1], l[:,2], l[:,3]

    gs_u, gs_c = np.unique(gs, return_counts=True ) 
    np.all( np.arange( len(gs_u) ) == gs_u )       

    id_u, id_c = np.unique( id_, return_counts=True  )  
    assert np.all( id_c == 1 )  
    ix_u, ix_c = np.unique( ix, return_counts=True )  

    gn_u, gn_c = np.unique( gn, return_counts=True )  
    print(gn_u)
    print(gn_c)


if __name__ == '__main__':

    t = Fold.Load() 
    PIDX = int(os.environ.get("PIDX","-1"))
    print(t)

    # pho: labels are collected within U4Recorder::PreUserTrackingAction 
    l = t.pho if hasattr(t, "pho") else None      # labels slotted in using spho::id
    check_pho_labels(l)

    gs, ix, id_, gn = l[:,0], l[:,1], l[:,2], l[:,3] 


    p = t.photon if hasattr(t, "photon") else None
    r = t.record if hasattr(t, "record") else None
    seq = t.seq if hasattr(t, "seq") else None
    nib = seqnib_(seq[:,0])  if not seq is None else None

    for i in range(len(p)):
        if not (PIDX == -1 or PIDX == i): continue 
        if PIDX > -1: print("PIDX %d " % PIDX) 
        print("r[%d,:,:3]" % i)
        print(r[i,:nib[i],:3]) 
        print("\n\nbflagdesc_(r[%d,j])" % i)
        for j in range(nib[i]):
            print(bflagdesc_(r[i,j]))   
        pass

        #print("ridiff_(r[%d])*1000." % i)
        #print(ridiff_(r[i])*1000.)   

        print("\n") 
        print("p[%d]" % i)
        print(p[i])
        print("\n") 
        print("bflagdesc_(p[%d])" % i)
        print(bflagdesc_(p[i])) 
        print("\n") 
        if not seq is None:
            print("seqhis_(seq[%d,0]) nib[%d]  " % (i,i) ) 
            print(" %s : %s "% ( seqhis_(seq[i,0]), nib[i] ))
            print("\n")
        pass
        print("\n\n")
    pass
    idx = p.view(np.uint32)[:,3,2] 
    assert np.all( np.arange( len(p) ) == idx ) 

    flagmask_u, flagmask_c = np.unique(p.view(np.uint32)[:,3,3], return_counts=True)    
    print("flagmask_u:%s " % str(flagmask_u))
    print("flagmask_c:%s " % str(flagmask_c))

    #print("\n".join(seqhis_( t.seq[:,0] ))) 
    for i in range(min(100,len(t.seq))):
        print("%4d : %s " % (i, seqhis_(t.seq[i,0])))
    pass
pass

