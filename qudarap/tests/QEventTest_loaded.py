#!/usr/bin/env python

import os, numpy as np

if __name__ == '__main__':
    TEST = os.environ["TEST"]
    assert TEST == "loaded"
    p_ = "$TMP/sysrap/SEventTest/cegs.npy"
    p = os.path.expandvars(p_)
    a = np.load(p)

    print("TEST:%s" % TEST)
    print("p:%s" % p )
    print("a.shape\n", a.shape )

    e = "a[:,0].view(np.int32)"
    print(e)
    print(eval(e))


    


