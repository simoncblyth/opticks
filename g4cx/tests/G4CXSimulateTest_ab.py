#!/usr/bin/env python 
"""
G4CXSimulateTest_ab.py 
======================

Usage::

   gx
   ./gxs_ab.sh  

"""
import numpy as np

from opticks.ana.fold import Fold
from opticks.ana.p import * 
from opticks.ana.eprint import eprint, epr

from opticks.sysrap.xfold import XFold
from opticks.sysrap.ABR import ABR 
from opticks.sysrap.stag import stag  
from opticks.u4.U4Stack import U4Stack

np.set_printoptions(edgeitems=16) 

tag = stag()
stack = U4Stack()
SLOTS = stag.SLOTS   



if __name__ == '__main__':

    quiet = True 
    a = Fold.Load("$A_FOLD", symbol="a", quiet=quiet) if "A_FOLD" in os.environ else None
    b = Fold.Load("$B_FOLD", symbol="b", quiet=quiet) if "B_FOLD" in os.environ else None
    ab = (not a is None) and (not b is None)

    print("-------- after Fold.Load" )

    A = XFold(a, symbol="A") if not a is None else None
    B = XFold(b, symbol="B") if not b is None else None
    AB = ABR(A,B) if ab else None 

    print("-------- after XFold" )

    if ab: 
        im = epr("im = np.abs(a.inphoton - b.inphoton).max()", globals(), locals() )  
        pm = epr("pm = np.abs(a.photon - b.photon).max()",     globals(), locals() )  
        rm = epr("rm = np.abs(a.record - b.record).max()",     globals(), locals() )  
        sm = epr("sm = np.all( a.seq[:,0] == b.seq[:,0] )",    globals(), locals() ) 

        we_ = "we = np.where( A.t.view('|S%(SLOTS)s') != B.t2.view('|S%(SLOTS)s') )[0]" % locals()  # eg stag.h/stag::SLOTS = 64 
        we = epr(we_,  globals(), locals() ) 

        eprint("np.all( A.ts == B.ts2 )", globals(), locals() )
        eprint("np.all( A.ts2 == B.ts )", globals(), locals() )

        assert (a.inphoton - b.inphoton).max() < 1e-5 
        #assert np.all( A.ts == B.ts2 ) 
        #assert np.all( A.ts2 == B.ts )  
    pass

    print("./U4RecorderTest_ab.sh ## u4t ")
    w = epr("w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0])", globals(), locals() )
    s = epr("s = a.seq[w,0]", globals(), locals() )  
    o = epr("o = cuss(s,w)", globals(), locals() , rprefix="\n")

    #epr("w1", globals(), locals() )
    #abw0 = epr("abw0 = a.photon[w0,:4] - b.photon[w0,:4]", globals(), locals(), rprefix="\n" ) 

    epr("a.base", globals(), locals() )
    epr("b.base", globals(), locals() )





