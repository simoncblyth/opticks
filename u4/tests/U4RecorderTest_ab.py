#!/usr/bin/env python 
"""
U4RecorderTest_ab.py
======================

Usage::

   cd ~/opticks/u4/tests   # u4t
   ./U4RecorderTest_ab.sh  

"""
import numpy as np

from opticks.ana.fold import Fold
from opticks.ana.p import * 
from opticks.ana.eprint import eprint, epr

from opticks.sysrap.xfold import XFold
from opticks.sysrap.stag import stag  
from opticks.u4.U4Stack import U4Stack

np.set_printoptions(edgeitems=16) 

tag = stag()
stack = U4Stack()


if __name__ == '__main__':

    a = Fold.Load("$A_FOLD", symbol="a") if "A_FOLD" in os.environ else None
    b = Fold.Load("$B_FOLD", symbol="b") if "B_FOLD" in os.environ else None

    A = XFold(a, symbol="A") if not a is None else None
    B = XFold(b, symbol="B") if not b is None else None

    ab = (not a is None) and (not b is None)
    if ab: 
        im = epr("im = np.abs(a.inphoton - b.inphoton).max()", globals(), locals() )  
        pm = epr("pm = np.abs(a.photon - b.photon).max()",     globals(), locals() )  
        rm = epr("rm = np.abs(a.record - b.record).max()",     globals(), locals() )  
        sm = epr("sm = np.all( a.seq[:,0] == b.seq[:,0] )",    globals(), locals() )  

        eprint("np.all( A.ts == B.ts2 )", globals(), locals() )
        eprint("np.all( A.ts2 == B.ts )", globals(), locals() )

        assert (a.inphoton - b.inphoton).max() < 1e-5 
        #assert np.all( A.ts == B.ts2 ) 
        #assert np.all( A.ts2 == B.ts )  
    pass

    w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0])
    s = a.seq[w,0]  
    o = cuss(s,w)
    print(o)
    print(w1)




