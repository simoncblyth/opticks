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

from opticks.sysrap.stag import stag  
from opticks.u4.U4Stack import U4Stack
stack = U4Stack()


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


def parse_slice(ekey):
    if not ekey in os.environ: return slice(None)
    elem = os.environ[ekey].split(":")    
    start = None if elem[0] == "" else int(elem[0])
    stop = None if elem[1] == "" else int(elem[1])
    step = None if elem[2] == "" else int(elem[2])
    return slice(start, stop, step) 



if __name__ == '__main__':

    t = Fold.Load() 
    PIDX = int(os.environ.get("PIDX","-1"))
    print(t)

    if "RECS_PLOT" in os.environ:
        """
        RECS_PLOT=1 ./U4RecorderTest.sh ana 


        RECS_PLOT=1 SEQHIS="TO BT BT SA" RSLI="0::4" ./U4RecorderTest.sh ana 
            select photons with one history and then just plot the first record step 

        RECS_PLOT=1 SEQHIS="TO BT BT SA" RSLI="1::4" ./U4RecorderTest.sh ana 
            expected to show positions of first BT

            np.all(np.tile(np.arange(4),100)[slice(0,None,4)]==0)    
            np.all(np.tile(np.arange(4),100)[slice(1,None,4)]==1)    
            np.all(np.tile(np.arange(4),100)[slice(2,None,4)]==2)  
            np.all(np.tile(np.arange(4),100)[slice(3,None,4)]==3)  

        RECS_PLOT=1 SEQHIS="TO BT BT SA" RSLI="2::4" ./U4RecorderTest.sh ana 

        RECS_PLOT=1 SEQHIS="TO BT BT SA" RSLI="3::4" ./U4RecorderTest.sh ana 

        RECS_PLOT=1 SEQHIS="TO BT BT SA" IREC=2 ./U4RecorderTest.sh ana 


        RECS_PLOT=1 SEQHIS="TO BT BT BT BT BT SA" IREC=1 ./U4RecorderTest.sh ana 

        RECS_PLOT=1 SEQHIS="TO BT BT BT BT BT SA" ./U4RecorderTest.sh ana 
             plot all step points 

        """
        from opticks.ana.pvplt import * 
        pl = pvplt_plotter(label="dump all record point positions")      
        if "SEQHIS" in os.environ: 
            #w_sel = np.where( t.seq[:,0] == cseqhis_("TO BT BT SA") )[0]
            w_sel = np.where( t.seq[:,0] == cseqhis_(os.environ["SEQHIS"]) )[0]
        else:
            w_sel = slice(None)
        pass 

        if "RSLI" in os.environ:
            ## This way of selecting record points is overly complicated, much better 
            ## to use the more direct IREC approach. 
            w_rec = np.where( t.record[w_sel,:,3,3].view(np.int32) != 0 )
            #recs = t.record[w_rec]   ## flattens step points to give shape eg (35152, 4, 4) where 8788*4  = 35152 
            ## the above would be an error as it is using w_sel sub-selection indices as if they were full array indices. 
            recs = t.record[w_sel][w_rec]
            ## To correctly use a where selection the base selection must be the same as that 
            ## used to create the where selection. 
            rsli = parse_slice("RSLI")            
            rpos = recs[rsli,0,:3]   
        elif "IREC" in os.environ:
            ## using IREC is a simpler more direct way of selecting step points  
            irec = int(os.environ["IREC"])
            rpos = t.record[w_sel, irec, 0,:3]
        else:
            w_rec = np.where( t.record[w_sel,:,3,3].view(np.int32) != 0 )   
            ## Above w_rec skips unfilled record points.  
            ## Notice that for w_sel slice(None) for example w_rec is selecting 
            ## different numbers of step points for each photon : so the below rpos contains 
            ## all positions from all step points within the w_sel selection. This includes 
            ## bulk scatter and absorption points. 
            ## TODO: find way to do similar but select based on the flag of the step points 
            ## so can color plots according to the flag. 
            rpos = t.record[w_sel][w_rec][:,0,:3]
        pass

        pl.add_points(rpos)   
        pl.show()
    pass




    # pho: labels are collected within U4Recorder::PreUserTrackingAction 
    l = t.pho if hasattr(t, "pho") else None      # labels slotted in using spho::id
    check_pho_labels(l)

    gs, ix, id_, gn = l[:,0], l[:,1], l[:,2], l[:,3] 

    st = stag.Unpack(t.tag) if hasattr(t,"tag") else None

    p = t.photon if hasattr(t, "photon") else None
    r = t.record if hasattr(t, "record") else None
    seq = t.seq if hasattr(t, "seq") else None
    nib = seqnib_(seq[:,0])  if not seq is None else None

    lim = min(len(p), 3)
    for i in range(lim):
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
    for i in range(lim):
        print("%4d : %s " % (i, seqhis_(t.seq[i,0])))
    pass

    seq_u, seq_c = np.unique(seq[:,0], return_counts=True)  
    for i in range(len(seq_u)):
        print("%4d : %4d : %20s  " % (i, seq_c[i], seqhis_(seq_u[i]) ))
    pass
    print("t.base : %s " %  t.base)
pass

