#!/usr/bin/env python

"""
QCerenkovTest.py
================

::

    QCerenkovTest 
    ipython -i tests/QCerenkovTest.py

"""
import os, logging, numpy as np
log = logging.getLogger(__name__)
from opticks.ana.nload import stamp_


class QCerenkovTest(object):
    BASE = os.path.expandvars("/tmp/$USER/opticks/QCerenkovTest") 
    def __init__(self):
        pass

    def test_lookup(self):
        fold = os.path.join(self.BASE, "test_lookup")
        names = os.listdir(fold)
        for name in filter(lambda n:n.endswith(".npy"),names):
            path = os.path.join(fold, name)
            stem = name[:-4]
            a = np.load(path)
            log.info(" %10s : %20s : %s : %s " % ( stem, str(a.shape), stamp_(path), path )) 
            setattr( self, stem, a ) 
            globals()[stem] = a 
        pass
        assert np.all( icdf_src == icdf_dst )  


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    t = QCerenkovTest()
    t.test_lookup()

    


