#!/usr/bin/env python
"""
::

    ipython -i tests/QTexLookupTest.py 

"""
import os, logging, numpy as np
log = logging.getLogger(__name__)


class QTexLookupTest(object):
    FOLD = os.path.expandvars("/tmp/$USER/opticks/QTexLookupTest") 

    def __init__(self, fold=None):
        if fold is None:
            fold = self.FOLD
        pass        
        names = os.listdir(fold)
        log.info("loading from fold %s " % fold)
        for name in filter(lambda _:_.endswith(".npy"), names):
            path = os.path.join(fold, name)
            stem = name[:-4]
            a = np.load(path) 
            print( " t.%5s  %s " % (stem, str(a.shape))) 
            setattr(self, stem, a )
            globals()[stem] = a 
        pass
        self.fold = fold
    pass

if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)
     t = QTexLookupTest()

     np.all( t.lookup == t.origin )  


