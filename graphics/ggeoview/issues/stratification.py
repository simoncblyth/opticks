#!/usr/bin/env python
"""
See http://localhost/env_notes/graphics/ggeoview/issues/stratification/

"""
import os, logging
import numpy as np
from env.python.utils import *
from env.numerics.npy.types import *
import env.numerics.npy.PropLib as PropLib 

log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from env.numerics.npy.ana import Evt, Selection, Rat, theta

np.set_printoptions(suppress=True, precision=3)

def scatter3d(fig,  xyz): 
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])

def histo(fig,  vals): 
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=91,range=[0,90])

rat_ = lambda n,d:float(len(n))/float(len(d))



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    e = Evt(tag="1")

    a = Selection(e)
    s = Selection(e,"BT SA")

    i = s.recpos(1)
    z = i[:,2]

    

    #p0a = a.recpos(0) 
 
    #p0 = s.recpos(0)
    #p1 = s.recpos(1)
    #p2 = s.recpos(2)




