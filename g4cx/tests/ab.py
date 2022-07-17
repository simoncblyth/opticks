#!/usr/bin/env python 
"""
G4CXSimulateTest_ab.py 
======================

Usage::

   gx
   ./ab.sh  


    In [15]: np.all( A.ts == B.ts2[:,:12,:29] )                                                                                            
    Out[15]: False

    In [17]: A.ts.shape                                                                                                                    
    Out[17]: (10000, 12, 29)

    In [18]: B.ts2.shape                                                                                                                   
    Out[18]: (10000, 13, 29)


Warning from comparing different shapes::

    In [16]: np.all( A.ts == B.ts2 )                                                                                                       
    /Users/blyth/miniconda3/bin/ipython:1: DeprecationWarning: elementwise comparison failed; this will raise an error in the future.
      #!/Users/blyth/miniconda3/bin/python
    Out[16]: False

::

    In [19]: np.where( A.ts != B.ts2[:,:12,:29] )                                                                                          
    Out[19]: 
    (array([9964, 9964, 9964, 9964, 9964, 9964]),
     array([ 2,  4,  6,  8, 10, 11]),
     array([5, 5, 5, 5, 5, 3]))


Mis-aligned from truncation ?::

    In [20]: seqhis_(a.seq[9964,0])                                                                                                        
    Out[20]: 'TO BT SC BR BR BR BR BR BR BR'

    In [21]: seqhis_(b.seq[9964,0])                                                                                                        
    Out[21]: 'TO BT SC BR BR BR BR BR BR BR'



"""
import numpy as np
import logging
log = logging.getLogger(__name__)

from opticks.ana.fold import Fold
from opticks.ana.p import * 
from opticks.ana.eprint import eprint, epr, edv

from opticks.sysrap.xfold import XFold
from opticks.sysrap.ABR import ABR 
from opticks.sysrap.stag import stag  
from opticks.u4.U4Stack import U4Stack

np.set_printoptions(edgeitems=16) 

tag = stag()
stack = U4Stack()
SLOTS = stag.SLOTS   


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    quiet = True 
    a = Fold.Load("$A_FOLD", symbol="a", quiet=quiet) if "A_FOLD" in os.environ else None
    b = Fold.Load("$B_FOLD", symbol="b", quiet=quiet) if "B_FOLD" in os.environ else None

    ## SUSPECT photon element (1,3) is starting uninitialized in input photon running
    ## forcing the below kludge to be able to compare 
    np.all( b.record[:,0,1,3] == 0. )                                                                                                    
    np.all( a.record[:,0,1,3] == 1. )                                                                                                    
    a.record[:,0,1,3] = 1.   # kludge
    b.record[:,0,1,3] = 1.   # kludge 

    ab = (not a is None) and (not b is None)

    print("-------- after Fold.Load" )

    A = XFold(a, symbol="A") if not a is None else None

    print("-------- after XFold.A" )

    B = XFold(b, symbol="B") if not b is None else None

    print("-------- after XFold.B" )

    AB = ABR(A,B) if ab else None 

    print("-------- after ABR " )

    if ab: 
        im = epr("im = np.abs(a.inphoton - b.inphoton).max()", globals(), locals() )  
        pm = epr("pm = np.abs(a.photon - b.photon).max()",     globals(), locals() )  
        rm = epr("rm = np.abs(a.record - b.record).max()",     globals(), locals() )  
        sm = epr("sm = np.all( a.seq[:,0] == b.seq[:,0] )",    globals(), locals() ) 
        ww = epr("ww = np.where( a.seq[:,0] != b.seq[:,0] )[0]",    globals(), locals() ) 

        we_ = "we = np.where( A.t.view('|S%(SLOTS)s') != B.t2.view('|S%(SLOTS)s') )[0]" % locals()  # eg stag.h/stag::SLOTS = 64 
        we = epr(we_,  globals(), locals() ) 

        wm_ = "wm = np.where( A.t.view('|S%(SLOTS)s') == B.t2.view('|S%(SLOTS)s') )[0]" % locals()  # eg stag.h/stag::SLOTS = 64 
        wm = epr(wm_,  globals(), locals() ) 


        wa = epr("wa = np.unique(np.where( np.abs(a.record - b.record ) > 0.05 )[0])", globals(), locals() ) 

        print("---0---")
        eprint("np.all( A.ts == B.ts2 )", globals(), locals() )
        print("---1---")
        eprint("np.all( A.ts2 == B.ts )", globals(), locals() )
        print("---2---")

        assert (a.inphoton - b.inphoton).max() < 1e-3
        print("---3---")


        epr("o = cuss(a.seq[:,0])",  globals(), locals(), rprefix="\n" ) 
        edv("a.record[w0,1,0,2] - b.record[w0,1,0,2] # point 1 z", globals(), locals())  
        edv("a.record[w3,1,0,2] - b.record[w3,1,0,2] # point 1 z", globals(), locals())  
    pass

    print("./U4RecorderTest_ab.sh ## u4t ")
    w = epr("w = np.unique(np.where( np.abs(a.photon - b.photon) > 0.1 )[0])", globals(), locals() )
    s = epr("s = a.seq[w,0]", globals(), locals() )  
    o = epr("o = cuss(s,w)", globals(), locals() , rprefix="\n")

    #epr("w1", globals(), locals() )
    #abw0 = epr("abw0 = a.photon[w0,:4] - b.photon[w0,:4]", globals(), locals(), rprefix="\n" ) 

    epr("a.base", globals(), locals() )
    epr("b.base", globals(), locals() )








