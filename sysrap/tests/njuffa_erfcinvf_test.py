#!/usr/bin/env python

import os, numpy as np
import matplotlib.pyplot as mp  
SIZE = np.array([1280, 720])
np.set_printoptions(precision=5)


class A(np.ndarray): 
    @classmethod 
    def Load(cls, path_, symbol="a"):
        path = os.path.expandvars(path_)
        if os.path.exists(path):
            a = np.load(path) 
            r = a.view(cls)
            r.label = "%s:%s" % (symbol, path_)
        else:
            print("FAILED TO LOAD : %s " % path)
            r = None
        pass
        return r 

       



if __name__ == '__main__':

    a = A.Load("$FOLD/njuffa_erfcinvf_test.npy", symbol="a")
    b = A.Load("/tmp/erfcinvf_Test/erfcinvf_Test_cu.npy", symbol="b")
    c = A.Load("/tmp/S4MTRandGaussQTest/a.npy", symbol="c")

    fig, ax = mp.subplots(figsize=SIZE/100.)
    if not a is None:ax.scatter( a[:,0], a[:,1], s=0.2, label=a.label )
    if not b is None:ax.scatter( b[:,0], b[:,1], s=0.2, label=b.label )
    if not c is None:ax.scatter( c[:,0], c[:,1], s=0.2, label=c.label )
    pass
    ax.legend()
    fig.show()
pass

