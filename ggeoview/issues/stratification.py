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

from env.numerics.npy.ana import Evt, Selection, Rat, theta, scatter3d

X,Y,Z,W = 0,1,2,3

np.set_printoptions(suppress=True, precision=3)

rat_ = lambda n,d:float(len(n))/float(len(d))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    e = Evt(tag="1")

    a = Selection(e)
    s = Selection(e,"BT SA")

    i = s.recpost(1)
    z = i[:,Z]
    t = i[:,W]


    #fig = plt.figure()
    #scatter3d(fig, i)
    #fig.show()


    #p0a = a.recpost(0) 
    #p0 = s.recpost(0)
    #p1 = s.recpost(1)
    #p2 = s.recpost(2)




