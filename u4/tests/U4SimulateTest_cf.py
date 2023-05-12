#!/usr/bin/env python
"""
U4SimulateTest_cf.py
========================

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

    #print(repr(a))
    #print(repr(b))

    amt = ModelTrigger_Debug.Create(a.f, symbol="amt", publish=False)  # publish:True crashing 
    bmt = ModelTrigger_Debug.Create(b.f, symbol="bmt", publish=False)  # publish:True crashing 
    if (amt is None) or (bmt is None):
        print(" missing ModelTrigger_Debug amt:%s bmt:%s " % ( amt, bmt) )
        IMPL = "NoMTD"
    else:
        assert( amt.IMPL == bmt.IMPL )
        IMPL = amt.IMPL
    pass


    check_sevt = True
    if check_sevt:
        aq_ = a.f.seq[:,0]    #  shape eg (1000, 2, 2)
        bq_ = b.f.seq[:,0]     
        assert np.all( aq_ == a.q_ )
        assert np.all( bq_ == b.q_ )

        an = np.sum( seqnib_(aq_), axis=1 )     ## occupied nibbles across both sets of 16 from the two 64 bit ints 
        bn = np.sum( seqnib_(bq_), axis=1 )   
        assert np.all( an == a.n ) 
        assert np.all( bn == b.n ) 

        aq = ht.seqhis(aq_)  # "|S96"  32 point slots * 3 chars for each abbr eg "BT " 
        bq = ht.seqhis(bq_) 

        assert np.all( aq == a.q)
        assert np.all( bq == b.q)

        if 'SPECS' in os.environ:
            a_st_ = a.f.aux[:,:,2,3].view(np.int32)
            a_st = a.SPECS[a_st_]

            b_st_ = b.f.aux[:,:,2,3].view(np.int32)
            b_st = b.SPECS[b_st_]

            assert np.all( a.spec == a_st )
            assert np.all( b.spec == b_st )
        pass
    pass



    PID_DESC = "Dumping PID history and step specs with record position, time"
    if APID > -1:
        print("APID:%d # %s " % (APID,PID_DESC)  )
        exprs = "a.q[APID] np.c_[a.spec[APID,:a.n[APID]]] a.f.record[APID,:a.n[APID],0]"
        for expr in exprs.split(): 
            print(expr)
            print(repr(eval(expr)))
            print(".") 
        pass
    pass 
    if BPID > -1:
        print("BPID:%d # %s " % (BPID, PID_DESC) )
        exprs = "b.q[BPID] np.c_[b.spec[APID,:b.n[BPID]]] b.f.record[BPID,:b.n[BPID],0]"
        for expr in exprs.split(): 
            print(expr)
            print(repr(eval(expr))) 
            print(".") 
        pass
    pass 


    assert( a.CHECK == b.CHECK )
    CHECK = a.CHECK

    assert( a.LAYOUT == b.LAYOUT )
    LAYOUT = a.LAYOUT

    HEADLINE = "./U4SimulateTest.sh cf ## PMT Geometry : A(N=0) Unnatural+FastSim, B(N=1) Natural+CustomBoundary  "
    print("\n%s" % HEADLINE)
    print("GEOM/GEOMList/IMPL/LAYOUT/CHECK : %s/%s/%s/%s/%s " % (GEOM, GEOMList, IMPL, LAYOUT, CHECK) )

    qcf = QCF(a.q, b.q, symbol="qcf")

    print(qcf.aqu)
    print(qcf.bqu)
    print(qcf)

    #qcf0 = QCFZero(qcf)
    #print(qcf0)


