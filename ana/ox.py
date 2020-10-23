#!/usr/bin/env python
"""
ox.py : quick checks on photons
===================================

::

    In [2]: ox.view(np.int32)[:,3]                                                                                                                                                                 
    Out[2]: 
    array([[ -24,   -1,    0, 6400],
           [  41,   -1,    1, 6152],
           [  21,   -1,    2, 6656],
           ...,
           [  20,   -1, 9997, 6272],
           [  20,   -1, 9998, 6272],
           [  41,   -1, 9999, 6152]], dtype=int32)


    In [5]: np.all( ox.view(np.int32)[:,3,2] == np.arange(10000) )
    Out[5]: True


    In [10]: ox_flags[np.where(ox_flags[:,3] & hismask.code("SD"))]                                                                                                                                           
    Out[10]: 
    array([[ -30,  130,   19, 6208],
           [ -30,  139,   74, 6208],
           [ -30,  188,  217, 6208],
           [ -30,   76,  406, 6224],
           [ -30,  189,  546, 6208],
           [ -30,  181,  586, 7232],
           [ -30,  167,  690, 6208],
           [ -30,  185,  767, 6208],
             bnd sen-idx ox-idx hismask(OR of step flags)


"""

import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.histype import HisType
from opticks.ana.mattype import MatType
from opticks.ana.hismask import HisMask
from opticks.ana.blib import BLib
from opticks.ana.ggeo import GGeo

histype = HisType()
mattype = MatType()
hismask = HisMask() 
blib = BLib()
ggeo = GGeo()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        os.chdir(sys.argv[1])
        log.info("chdir %s " % os.getcwd())
    pass
    np.set_printoptions(suppress=True, linewidth=200)

    ox = np.load("ox.npy")
    ph = np.load("ph.npy") # seqhis, seqmat sequence histories for all photons
    seqhis = ph[:,0,0]
    seqmat = ph[:,0,1]

    ox_flags = ox.view(np.int32)[:,3]   
    ox_lander = ox_flags[ox_flags[:,1] != -1]

    print("ox_flags : %s " % repr(ox_flags.shape) )
    print("ox_lander : %s : photons landing on sensor volumes  " % repr(ox_lander.shape))  

    bndidx = np.unique(np.abs(ox[:,3,0].view(np.int32))) - 1   # subtract 1 to get index as signed boundaries are 1-based 
    print("bndidx : %s " % repr(bndidx))
    for _ in bndidx:print(blib.bname(_)) 

    for i, oxr in enumerate(ox):
        oxf = oxr[3].view(np.int32)

        ## use stomped on weight, for the "always there, not just sensors" node index of last intersected volume 
        nidx = oxr[1,3].view(np.uint32)  
        nrpo = ggeo.get_triplet_index(nidx)
        nidx2,ridx,pidx,oidx = nrpo
        assert nidx2 == nidx 
        #if ridx > 0: continue   # skip photons with last intersect on instanced geometry 
        if ridx == 0: continue   # skip photons with last intersect on remainder geometry 

        bnd,sidx,idx,pflg  = oxf
        sqh = seqhis[idx]
        sqm = seqmat[idx]
        msk = " %15s " % hismask.label(pflg) 
        his = "( %16x : %30s ) " % (sqh, histype.label(sqh)) 
        mat = "( %16x : %30s ) " % (sqm, mattype.label(sqm))
        print(" %5d : %15s :  %s %s %s : %s " % (i, oxf, msk,his,mat, nrpo) )
    pass
         
