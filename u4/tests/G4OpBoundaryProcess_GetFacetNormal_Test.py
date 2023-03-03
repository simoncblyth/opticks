#!/usr/bin/env python

import numpy as np
from opticks.ana.fold import Fold

np.set_printoptions(precision=12) 

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))
    print(t.FacetNormal[:100])

    fn = t.FacetNormal[:,0]
    sm = t.FacetNormal[:,1]
    mo = t.Meta[0] 

    fn_mo = np.sum( fn*mo, axis=1 )  # dot product of each FacetNormal with the mom 
    assert( np.all( fn_mo < 0 ) )    # all -ve so the mom is against the FacetNormal 

pass


