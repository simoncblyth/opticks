#!/usr/bin/env python

import os, numpy as np

FOLD = os.path.expandvars("/tmp/$USER/opticks/QSimTest")

if __name__ == '__main__':
    path_ = lambda ver:os.path.join(FOLD, "fill_state_%d.npy" % ver )

    p0 = path_(0)
    p1 = path_(1)

    s0 = np.load(p0)
    s1 = np.load(p1)

    print("  s0 %s p0 %s " % ( str(s0.shape), p0))
    print("  s1 %s p1 %s " % ( str(s1.shape), p1))

    assert np.all( s0.view(np.int32) == s1.view(np.int32) )

    # have to view as int to get the comparison to match as in float get non-comparable nan



    
