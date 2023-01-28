#!/usr/bin/env python
"""
stree_load_test_csg.py 
=======================


::

    In [15]: label,n[n[:,lv]==60]
    Out[15]: 
    ('        ix   dp   sx   pt   nc   fc   ns   lv   tc   pa   bb   xf',
     array([[242,   2,   0, 244,   0,  -1, 243,  60, 110, 151, 151,  -1],
            [243,   2,   1, 244,   0,  -1,  -1,  60, 110, 152, 152,  89],
            [244,   1,   0, 246,   2, 242, 245,  60,   1,  -1,  -1,  -1],
            [245,   1,   1, 246,   0,  -1,  -1,  60, 110, 153, 153,  90],
            [246,   0,  -1,  -1,   2, 244,  -1,  60,   1,  -1,  -1,  -1]], dtype=int32))


    int index ;        // ix
    int depth ;        // dp 
    int sibdex ;       // sx 
    int parent ;       // pt 

    int num_child ;    // nc
    int first_child ;  // fc 
    int next_sibling ; // ns
    int lvid ;         // lv

    int typecode ;     // tc
    int param ;        // pa
    int aabb ;         // bb
    int xform ;        // xf 

"""

import numpy as np
import builtins
from opticks.ana.fold import Fold
from opticks.sysrap.stree import stree

np.set_printoptions(edgeitems=16) 



if __name__ == '__main__':

    f = Fold.Load(symbol="f")
    print(repr(f))

    n = f.node

    fields = "ix dp sx pt nc fc ns lv tc pa bb xf".split()
    for i, fi in enumerate(fields): setattr(builtins, fi, i)
    label = " " * 8 + "   ".join(fields)
    print(label)
   

    assert np.all( np.arange(len(n)) == n[:,ix] )  
    assert np.all( n[:,dp] > -1 )    
    assert np.all( n[:,dp] < 10 )    


     



