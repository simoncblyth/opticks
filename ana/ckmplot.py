#!/usr/bin/env python
"""
::

    In [14]: evt.rpost_(slice(0,5)).shape
    Out[14]: (500000, 5, 4)


"""
import os, sys, re, logging, numpy as np
from collections import OrderedDict as odict

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main, gcp_
from opticks.ana.evt import Evt



def plot(evt, fig, rec=True, ox=False, mesh=True):
    """
    This will not work for millions of photons like Opticks geometry shaders do
    but handy anyway for an alternative interface to the record points 
    of a few hundred photons
    """
    ax = fig.gca(projection='3d')

    if mesh:
        ptn = re.compile("\d+")  
        meshidxs = map(int, filter(lambda n:ptn.match(n), os.listdir(gcp_("GMeshLib"))))
        for idx in meshidxs:
            vtx = np.load(gcp_("GMeshLib/%d/vertices.npy" % idx))
            ax.scatter( vtx[:,0], vtx[:,1], vtx[:,2]) 
        pass
    pass

    if rec:
        for i in range(min(300, len(evt.seqhis))):
            xyzt = evt.rposti(i) 
            x = xyzt[:,0]
            y = xyzt[:,1]
            z = xyzt[:,2]
            ax.plot(x, y, z )
        pass
    pass

    if ox:
        xs = evt.ox[:,0,0]
        ys = evt.ox[:,0,1]
        zs = evt.ox[:,0,2]
        ax.scatter( xs, ys, zs) 
    pass



if __name__ == '__main__':
    args = opticks_main(tag="1",src="natural", det="g4live", doc=__doc__)
    np.set_printoptions(suppress=True, precision=3)

    evt = Evt(tag=args.tag, src=args.src, det=args.det, seqs=[], args=args)

    log.debug("evt") 
    print(evt)

    log.debug("evt.history_table") 
    evt.history_table(slice(0,20))
    log.debug("evt.history_table DONE") 


    plt.ion()
    fig = plt.figure()

    plot(evt, fig)

    plt.show() 

    


       

