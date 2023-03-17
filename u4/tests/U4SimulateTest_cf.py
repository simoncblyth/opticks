#!/usr/bin/env python
"""
U4SimulateTest_cf.py
========================

Dump step point info on two single photons from A and B::

    POM=1 N=0 APID=475 BPID=476 ./U4SimulateTest.sh cf

"""


import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.p import * 
from opticks.ana.nbase import chi2  
from opticks.u4.tests.ModelTrigger_Debug import ModelTrigger_Debug       
from opticks.u4.tests.U4SimulateTest import U4SimulateTest

hist_ = lambda _:_.strip().decode("utf-8")   


NOGUI = "NOGUI" in os.environ
MODE = int(os.environ.get("MODE", 0))
if not NOGUI:
    from opticks.ana.pvplt import * 
pass

APID = int(os.environ.get("APID", -1))
BPID = int(os.environ.get("BPID", -1))

GEOM = os.environ.get("GEOM", "DummyGEOM")
GEOMList = os.environ.get("%s_GEOMList" % GEOM, "DummyGEOMList") 


if __name__ == '__main__':

    print("APID:%d" % (APID))
    print("BPID:%d" % (BPID))

    a = U4SimulateTest.Load("$AFOLD", symbol="a")
    b = U4SimulateTest.Load("$BFOLD", symbol="b")

    amt = ModelTrigger_Debug(a.f, symbol="amt", publish=False)  # publish:True crashing 
    bmt = ModelTrigger_Debug(b.f, symbol="bmt", publish=False)  # publish:True crashing 
    assert( amt.IMPL == bmt.IMPL )
    IMPL = amt.IMPL



    #print(repr(a))
    #print(repr(b))

    if APID > -1 and BPID > -1:
        print("seqhis_(a.f.seq[APID,0] : %s " % seqhis_(a.f.seq[APID,0] ))
        print("seqhis_(b.f.seq[APID,0] : %s " % seqhis_(b.f.seq[BPID,0] ))
        ar = a.f.record[APID]
        br = b.f.record[BPID]
        # mapping from new to old point index for PID 726 big bouncer
        b2a = np.array([ 0,1,3,5,6,8,9,11,12,13,15,17,19 ])
        abr = np.c_[ar[b2a,0],br[:len(b2a),0]].reshape(-1,2,4)
    pass

    lim = slice(0,10)

    aq_ = a.f.seq[:,0]    #  shape eg (1000, 2, 2)                                                                                                                  
    bq_ = b.f.seq[:,0]     

    an = np.sum( seqnib_(aq_), axis=1 )     ## occupied nibbles across both sets of 16 from the two 64 bit ints 
    bn = np.sum( seqnib_(bq_), axis=1 )   

    aq = ht.seqhis(aq_)  # "|S96"  32 point slots * 3 chars for each abbr eg "BT " 
    bq = ht.seqhis(bq_) 


    ## HMM: NEED AB OBJECT ?

    a_CHECK = a.f.photon_meta.CHECK[0]  
    b_CHECK = b.f.photon_meta.CHECK[0]  
    assert( a_CHECK == b_CHECK )
    CHECK = a_CHECK

    a_LAYOUT = a.f.photon_meta.LAYOUT[0]  
    b_LAYOUT = b.f.photon_meta.LAYOUT[0]  
    assert( a_LAYOUT == b_LAYOUT )
    LAYOUT = a_LAYOUT


    a_SPECS = np.array(a.f.U4R_names.lines)
    a_st_ = a.f.aux[:,:,2,3].view(np.int32)
    a_st = a_SPECS[a_st_]

    b_SPECS = np.array(b.f.U4R_names.lines)
    b_st_ = b.f.aux[:,:,2,3].view(np.int32)
    b_st = b_SPECS[b_st_]


    PID_DESC = "Dumping PID history and step specs with record position, time"
    if APID > -1:
        print("APID:%d # %s " % (APID,PID_DESC)  )
        exprs = "aq[APID] np.c_[a_st[APID,:an[APID]]] a.f.record[APID,:an[APID],0]"
        for expr in exprs.split(): 
            print(expr)
            print(repr(eval(expr)))
            print(".") 
        pass
    pass 
    if BPID > -1:
        print("BPID:%d # %s " % (BPID, PID_DESC) )
        exprs = "bq[BPID] np.c_[b_st[APID,:bn[BPID]]] b.f.record[BPID,:bn[BPID],0]"
        for expr in exprs.split(): 
            print(expr)
            print(repr(eval(expr))) 
            print(".") 
        pass
    pass 






    ## TODO: rearrange the below into an ABQ object ? taking aq,bq as inputs  
    ## NOTICE ADVANTAGE OF NO DEPS APPROACH : COMPARED TO THE OLD AB MACHINERY 

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

    qu = np.unique(np.concatenate([aqu,bqu]))       ## unique histories of both A and B in uncontrolled order
    ab = np.zeros( (len(qu),3,2), dtype=np.int64 )

    for i, q in enumerate(qu):
        ai_ = np.where(aqu == q )[0]           # find indices in the a and b unique lists 
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
    iq = np.arange(len(qu)) 

    # more than 10 counts in one, but zero in the other : history dropouts are smoking guns for bugs 
    bzero = np.where( np.logical_and( abo[:,2,0] > 10, abo[:,2,1] == 0 ) )[0]                                                                                             
    azero = np.where( np.logical_and( abo[:,2,1] > 10, abo[:,2,0] == 0 ) )[0]                                                                                             

    c2cut = int(os.environ.get("C2CUT","30")) 
    c2,c2n,c2c = chi2( abo[:,2,0], abo[:,2,1], cut=c2cut )   
    c2sum = c2.sum()
    c2per = c2sum/c2n 


    siq = list(map(lambda _:"%2d" % _ , iq ))  
    sc2 = list(map(lambda _:"%7.4f" % _, c2 ))   

    HEADLINE = "./U4SimulateTest.sh cf ## PMT Geometry : A(N=0) Unnatural+FastSim, B(N=1) Natural+CustomBoundary  "
    print("\n%s" % HEADLINE)
    print("GEOM/GEOMList/IMPL/LAYOUT/CHECK : %s/%s/%s/%s/%s " % (GEOM, GEOMList, IMPL, LAYOUT, CHECK) )
    print("c2sum : %10.4f c2n : %10.4f c2per: %10.4f  C2CUT: %4d " % ( c2sum, c2n, c2per, c2cut ))  

    sabo2 = list(map(lambda _:"%6d %6d" % tuple(_), abo[:,2,:])) 
    sabo1 = list(map(lambda _:"%6d %6d" % tuple(_), abo[:,1,:])) 

    _quo = list(map(hist_, quo)) 
    mxl = max(list(map(len, _quo)))   
    fmt = "%-" + str(mxl) + "s"  
    _quo = list(map(lambda _:fmt % _, _quo ))
    _quo = np.array( _quo )  
  
    #abexpr = "np.c_[quo,abo[:,2,:],abo[:,1,:]]"
    abexpr = "np.c_[siq,_quo,siq,sabo2,sc2,sabo1]"  
    subs = "[:25] [bzero] [azero]".split()
    descs = ["A-B history frequency chi2 comparison", "bzero: A histories not in B", "azero: B histories not in A" ] 

    for i in range(len(subs)):
        expr = "%s%s" % (abexpr, subs[i])
        print("\n%s  ## %s " % (expr, descs[i]) )
        print(eval(expr))  
    pass

    lim = slice(0,2)


    print("\nbzero : %s : A HIST NOT IN B (A(N=0) has extra BT until remove fakes)" % (str(bzero)))
    for _ in bzero:
        idxs = np.where( quo[_] == aq[:,0] )[0] 
        print("bzero quo[_]:%s len(idxs):%d idxs[lim]:%s " % ( hist_(quo[_]), len(idxs), str(idxs[lim])) )
        for idx in idxs[lim]:
            viz = "u4t ; N=0 APID=%d AOPT=idx ./U4SimtraceTest.sh ana" % idx
            print(viz)
        pass
        if len(idxs) > 0: print("")
    pass

    print("\nazero : %s : B HIST NOT IN A" % (str(azero)))
    for _ in azero:
        idxs = np.where( quo[_] == bq[:,0] )[0]
        print("azero quo[_]:%s len(idxs):%d idxs[lim]:%s " % ( hist_(quo[_]), len(idxs), str(idxs[lim])) )
        for idx in idxs[lim]:
            viz = "u4t ; N=1 BPID=%d BOPT=idx ./U4SimtraceTest.sh ana" % idx
            print(viz)
        pass
        if len(idxs) > 0: print("")
    pass






