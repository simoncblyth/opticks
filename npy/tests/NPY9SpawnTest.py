#!/usr/bin/env python
"""
Noticed that the NP .npy serialization
pads the header more than the NPY .npy does.  
"""
import numpy as np
FOLD = os.path.expandvars("$TMP/NPY9SpawnTest")

if __name__ == '__main__':
    names = os.listdir(FOLD)
    for name in filter(lambda _:_.endswith(".npy"), names):
        stem = name[:-4]
        path = os.path.join( FOLD, name)
        a = np.load(path)
        print(" %10s : %15s : %s " % ( stem, str(a.shape), name ))
        globals()[stem] = a 
    pass
    assert np.all( f0 == f1 )
    assert np.all( d0 == d1 )
    assert np.all( i0 == i1 )





