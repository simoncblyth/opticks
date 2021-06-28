#!/usr/bin/env python
"""
::

    ipython -i tests/X4PhysicsVectorTest.py 


"""
import os, numpy as np
import matplotlib.pyplot as plt

class X4PhysicsVectorTest(object):

    DIR = "$TMP/X4PhysicsVectorTest"
    def __init__(self):
        g4interpolate = np.load(os.path.expandvars(os.path.join(self.DIR, "g4interpolate.npy")))
        convert = np.load(os.path.expandvars(os.path.join(self.DIR, "convert.npy")))
        self.g = g4interpolate 
        self.c = convert 


if __name__ == '__main__':
    pvt = X4PhysicsVectorTest()

    g = pvt.g
    c = pvt.c
    fig, ax = plt.subplots()  
    ax.plot( g[:,0], g[:,1], label="g", ds="steps-mid" ) 
    ax.plot( c[:,0], c[:,1], label="c", ds="steps-mid" )  
    ax.legend()   
    fig.show() 

