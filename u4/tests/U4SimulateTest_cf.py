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
from opticks.ana.qcf import QU,QCF,QCFZero

from opticks.u4.tests.ModelTrigger_Debug import ModelTrigger_Debug       
from opticks.sysrap.sevt import SEvt

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

    a = SEvt.Load("$AFOLD", symbol="a")
    b = SEvt.Load("$BFOLD", symbol="b")

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


    HEADLINE = "./U4SimulateTest.sh cf ## PMT Geometry : A(N=0) Unnatural+FastSim, B(N=1) Natural+CustomBoundary  "
    print("\n%s" % HEADLINE)
    print("GEOM/GEOMList/IMPL/LAYOUT/CHECK : %s/%s/%s/%s/%s " % (GEOM, GEOMList, IMPL, LAYOUT, CHECK) )

    qcf = QCF(aq,bq, symbol="qcf")

    print(qcf.aqu)
    print(qcf.bqu)
    print(qcf)

    #qcf0 = QCFZero(qcf)
    #print(qcf0)


