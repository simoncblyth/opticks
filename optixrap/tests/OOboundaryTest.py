#!/usr/bin/env python
"""

::

   ipython -i $(optixrap-sdir)/tests/OOboundaryTest.py 


"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

def load(tag, name, reshape=None):
    dir_ = os.path.expandvars("$TMP/%s" % name)
    path = os.path.join(dir_,"%s.npy" % tag)
    a = np.load(path)
    log.info(" load %s %s %s " % (tag, repr(a.shape), path))
    if reshape is not None:
        a = a.reshape(reshape)
        log.info("reshape %s to %s " % (tag, repr(a.shape)))
    pass
    return a 

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(suppress=True, precision=3)

    #name = "OOboundaryTest" 
    name = "OOboundaryLookupTest" 

    ori = load("ori", name)
    inp = load("inp", name)
    out = load("out", name, reshape=ori.shape)

    if not np.allclose(ori, out):
        for i in range(len(ori)):
            ok = np.allclose(out[i],ori[i])
            print "%4d %s \n" % (i, ok)
        pass
    pass
    assert np.allclose(ori, out)







