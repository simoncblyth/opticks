#!/usr/bin/env python
"""
U4SimulateTest_cf.py
========================

::

    PID = 726    
    seqhis_(a.seq[PID,0] : ['TO BT BT BT BT SR SR BT BR BR BT SR SR SR BT BR', 'BT SR BT SA'] 
    seqhis_(b.seq[PID,0] : ['TO BT BT SR SR BR BR SR SR SR BR SR BR SR SA', '?0?'] 



"""


import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.p import * 

NOGUI = "NOGUI" in os.environ
MODE = int(os.environ.get("MODE", 0))
if not NOGUI:
    from opticks.ana.pvplt import * 
pass

PID = int(os.environ.get("PID", -1))
if PID == -1: PID = int(os.environ.get("OPTICKS_G4STATE_RERUN", -1))

if __name__ == '__main__':

    print("PID : %d " % (PID))
    a = Fold.Load("$AFOLD", symbol="a")
    b = Fold.Load("$BFOLD", symbol="b")

    #print(repr(a))
    #print(repr(b))

    if PID > -1:
        print("seqhis_(a.seq[PID,0] : %s " % seqhis_(a.seq[PID,0] ))
        print("seqhis_(b.seq[PID,0] : %s " % seqhis_(b.seq[PID,0] ))
        ar = a.record[PID]
        br = b.record[PID]
        # mapping from new to old point index for PID 726 big bouncer
        b2a = np.array([ 0,1,3,5,6,8,9,11,12,13,15,17,19 ])
        abr = np.c_[ar[b2a,0],br[:len(b2a),0]].reshape(-1,2,4)
    pass

    lim = slice(0,10)

    aq_ = a.seq[:,0]    #  shape eg (1000, 2, 2)                                                                                                                  
    bq_ = b.seq[:,0]     

    aq = ht.seqhis(aq_)  # "|S96"  32 point slots * 3 chars for each abbr eg "BT " 
    bq = ht.seqhis(bq_) 

    ## resort to uniqing the "|S96" label because NumPy lacks uint128 : so cannot hold the history in a single big int  
    aqu, aqi, aqn = np.unique(aq, return_index=True, return_counts=True)
    aquo = np.argsort(aqn)[::-1]  
    aexpr = "np.c_[aqn,aqi,aqu][aquo][lim]"

    bqu, bqi, bqn = np.unique(bq, return_index=True, return_counts=True)
    bquo = np.argsort(bqn)[::-1]  
    bexpr = "np.c_[bqn,bqi,bqu][bquo][lim]"
    
    print("\n%s  ## aexpr : unique histories aqu in descending count aqn order, aqi first index " % aexpr )
    print(eval(aexpr))  

    print("\n%s  ## bexpr : unique histories bqu in descending count bqn order, bqi first index " % bexpr )
    print(eval(bexpr))  


    qu = np.unique(np.concatenate([aqu,bqu])) ## unique histories of both A and B in uncontrolled order

    ab = np.zeros( (len(qu),3,2), dtype=np.int64 )

    for i, q in enumerate(qu):
        ai_ = np.where(aqu == q )[0]   # find indices in the a and b unique lists 
        bi_ = np.where(bqu == q )[0]   
        ai = ai_[0] if len(ai_) == 1 else -1
        bi = bi_[0] if len(bi_) == 1 else -1

        ab[i,0,0] = ai
        ab[i,1,0] = aqi[ai] if ai > -1 else -1 
        ab[i,2,0] = aqn[ai] if ai > -1 else 0 

        ab[i,0,1] = bi
        ab[i,1,1] = bqi[bi] if bi > -1 else -1 
        ab[i,2,1] = bqn[bi] if bi > -1 else 0 
    pass

    abx = np.max(ab[:,2,:], axis=1 )   # max of aqn, bqn counts 
    abxo = np.argsort(abx)[::-1]       # descending count order indices
    abo = ab[abxo]                     # ab ordered  
    quo = qu[abxo]                     # qu ordered 


    #abexpr = "np.c_[quo,abo[:,2,:],abo[:,1,:]][:30]"
    abexpr = "np.c_[np.arange(len(quo)),quo,np.arange(len(quo)),abo[:,2,:],abo[:,1,:]][:30]"  

    print("\n%s  ## abexpr : A-B comparison of unique history counts " % abexpr )
    print(eval(abexpr))  







