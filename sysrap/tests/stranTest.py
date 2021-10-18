#!/usr/bin/env python
"""
ipython -i tests/stranTest.py 

"""
import os, numpy as np

class stranTest(object):
    FOLD = os.path.expandvars("/tmp/$USER/opticks/sysrap/stranTest/test_write")
    def __init__(self, fold=FOLD):
        names = os.listdir(fold)
        for name in filter(lambda n:n.endswith(".npy"), names):
            path = os.path.join(fold,name)
            stem = name[:-4]
            a = np.load(path)
            print(" %6s : %15s : %s " % (stem, str(a.shape), path))
            globals()[stem] = a  
        pass


if __name__ == '__main__':
    t = stranTest()



