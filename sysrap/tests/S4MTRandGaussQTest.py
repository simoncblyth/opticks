#!/usr/bin/env python

import numpy as np
SIZE = np.array([1280, 720])
import matplotlib.pyplot as mp


def test_transformQuick():
    a = np.load(os.path.expandvars("$FOLD/test_transformQuick.npy"))
    fig, ax = mp.subplots(figsize=SIZE/100.)
    ax.scatter( a[:,0], a[:,1], s=0.1 )
    fig.show() 
    return a 


def test_transformQuick_vs_njuffa_erfcinvf():
    a = np.load(os.path.expandvars("$FOLD/test_transformQuick_vs_njuffa_erfcinvf.npy"))
    fig, ax = mp.subplots(figsize=SIZE/100.)
    ax.scatter( a[:,0], a[:,1], s=0.1, label="transformQuick" )
    ax.scatter( a[:,0], a[:,2], s=0.1, label="njuffa_erfcinvf" )
    ax.legend()
    fig.show() 
    return a


def test_shoot():
    a = np.load(os.path.expandvars("$FOLD/test_shoot.npy"))
    fig, ax = mp.subplots(figsize=SIZE/100.)
    ax.hist(a, bins=100)
    fig.show() 
    return a



if __name__ == '__main__':
    #a = test_shoot()
    a = test_transformQuick_vs_njuffa_erfcinvf()

pass
