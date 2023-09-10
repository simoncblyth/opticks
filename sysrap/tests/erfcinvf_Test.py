#!/usr/bin/env python
"""


"""
import os, numpy as np
import matplotlib.pyplot as mp  
SIZE = np.array([1280, 720])

np.set_printoptions(precision=5)


def load(path):
    if True and os.path.exists(path):
        print(path)
        a = np.load(path) 
    else:
        print("FAILED TO LOAD : %s " % path)
        a = None
    pass
    return a 



if __name__ == '__main__':


    a = load("/tmp/erfcinvf_Test/erfcinvf_Test_cu.npy")
    b = load("/tmp/S4MTRandGaussQTest/a.npy")

    if not a is None and not b is None:
        ab = b[:,1] - a[:,2]
    else:
        ab = None
    pass 

    fig, ax = mp.subplots(figsize=SIZE/100.)

    if not a is None:
        ax.scatter( a[:,0], a[:,1], s=0.2, label="a" )
    pass

    if not b is None:
        ax.scatter( b[:,0], b[:,1], s=0.2, label="b" )
    pass

    ax.legend()
    fig.show()



