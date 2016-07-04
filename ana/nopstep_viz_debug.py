#!/usr/bin/env python

import os, logging
import numpy as np

import matplotlib.pyplot as plt
from opticks.ana.nload import A
from opticks.ana.CGDMLDetector import CGDMLDetector

def make_fake_nopstep(tk):
    """
    :param tk: dict of dict containing track parameters
    :return nop:  numpy array of shape (tot n, 4, 4)
                  containing time and global positions in [:,0] 

    For each track

    *n* 
          number of time steps 
    *tmin*
          ns
    *tmax*
          ns
    *fn*
          parametric equation returning local coordinate [0,1,2] from time input           
    *frame*
          4x4 homogenous matrix applied to the local trajectory coordinates
          to get global coordinates

    """
    traj = {}
    ftraj = {}

    for k in tk.keys():

        tkd = tk[k]
        n = tkd["n"]
        fn = tkd["fn"]
        mat = tkd["frame"]

        traj[k] = np.ones([n, 4], dtype=np.float32)
        t = np.linspace(tkd["tmin"],tkd["tmax"], n)

        for p in range(n):
            traj[k][p,:3] = fn(t[p])    
        pass

        ftraj[k] = np.dot(traj[k], mat)   
        ftraj[k][:,3] = t 
    pass 

    combi = np.vstack(ftraj.values())
    nop = np.zeros([combi.shape[0], 4, 4], dtype=np.float32)
    nop[:,0] = combi 

    return nop



if __name__ == '__main__':

    frame = 3153

    det = CGDMLDetector()
    mat = det.getGlobalTransform(frame)
    print "mat %s " % repr(mat)

    tk = {} 
    tk["+x"] = dict(n=10,frame=mat,tmin=0,tmax=10,fn=lambda t:np.array([0,0,0]) + t*np.array([100,0,0]))
    tk["-x"] = dict(n=20,frame=mat,tmin=1,tmax=20,fn=lambda t:np.array([0,0,0]) + t*np.array([-100,0,0]))
    tk["+y"] = dict(n=10,frame=mat,tmin=0,tmax=10,fn=lambda t:np.array([0,0,0]) + t*np.array([0,100,0]))
    tk["-y"] = dict(n=20,frame=mat,tmin=1,tmax=20,fn=lambda t:np.array([0,0,0]) + t*np.array([0,-100,0]))
    tk["+z"] = dict(n=10,frame=mat,tmin=0,tmax=10,fn=lambda t:np.array([0,0,0]) + t*np.array([0,0,100]))
    tk["-z"] = dict(n=20,frame=mat,tmin=1,tmax=20,fn=lambda t:np.array([0,0,0]) + t*np.array([0,0,-100]))


    nop = make_fake_nopstep(tk)

    path = "/tmp/nopstep.npy"
    np.save(path, nop)
 
    a = np.load(path)


