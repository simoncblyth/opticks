#!/usr/bin/env python

import os, sys, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.eprint import eprint 

if __name__ == '__main__':
    t = Fold.Load(globals=True)
    print("\n\n")
    eprint("np.all( t.boundary_lookup_all == t.boundary_lookup_all_src )", globals(), locals() ) 

    print(sys.argv[0])
 
