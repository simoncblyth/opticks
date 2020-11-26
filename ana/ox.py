#!/usr/bin/env python
"""
ox.py : quick checks on photons
===================================

::

    In [36]: ox.view(np.int32)[:,3]                                                                                                                                                                                                                                                   
    Out[36]: 
    array([[-1507329,     3159,        0,     6400],
           [ 2752511,     4430,        1,     6152],
           [ 1441791,     4425,        2,     6656],
           ...,
           [ 1376255,     3155,     4997,     6272],
           [-1376257,     3157,     4998,     6416],
           [ 1376255,     3155,     4999,     6272]], dtype=int32)

           bnd_sidx
           two int16       nidx     phidx     flags

    In [37]: np.all( ox.view(np.int32)[:,3,2] == np.arange(5000) ) 
    Out[37]: True

    In [38]: ox_flags[np.where(ox_flags[:,3] & hismask.code("SD"))]                                                                                                                                                                                                                   
    Out[38]: 
    array([[-1965950,     3981,       19,     6208],
           [-1965941,     4035,       74,     6208],
           [-1965892,     4329,      217,     6208],
           [-1966004,     3657,      406,     6224],
           [-1965891,     4335,      546,     6208],
           [-1965899,     4287,      586,     7232],
           [-1965913,     4203,      690,     6208],

    In [41]: ox_flags[np.where(ox_flags[:,3] & hismask.code("SD"))][:,0] & 0xffff                                                                                                                                                                                                     
    Out[41]: 
    array([130, 139, 188,  76, 189, 181, 167, 185, 152, 150,  29,  89,  97, 160, 183,  37, 132,  50,  13, 169, 141,  84,  73,  85, 144, 128,  87,  19, 187, 174,  76, 180, 101,  82, 116,  66,  29,  63,
            88, 165,  36, 169,   9, 121,  23, 129, 143], dtype=int32)

    In [42]: ox_flags[np.where(ox_flags[:,3] & hismask.code("SD"))][:,0] >> 16                                                                                                                                                                                                        
    Out[42]: 
    array([-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,  30,  30,  30, -30, -30, -30, -30, -30, -30, -30, -30,  30, -30, -30,
           -30, -30, -30, -30,  30, -30, -30, -30, -30], dtype=int32)


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


def dump_boundaries(ox):
    bndidx = (ox[:,3,0].view(np.uint32) >> 16).view(np.int16)[0::2] 
    u_bndidx, u_bndidx_counts = np.unique(bndidx, return_counts=True)  
    tot = 0 
    print("dump_boundaries")
    for bnd,bnd_count in sorted(zip(u_bndidx,u_bndidx_counts), key=lambda _:_[1], reverse=True):
        name = blib.bname(np.abs(bnd)-1)  # subtract 1 to get index as signed boundaries are 1-based 
        print("%4d : %7d  : %s " % (bnd, bnd_count, name))
        tot += bnd_count
    pass
    print("%4s : %7d " % ("TOT",tot))

def dump_sensorIndex(ox):
    sidx = (ox[:,3,0].view(np.uint32) & 0xffff).view(np.int16)[0::2] 
    u_sidx, u_sidx_counts = np.unique(sidx, return_counts=True)  
    tot = 0 
    print("dump_sensorIndex")
    for sid,sid_count in sorted(zip(u_sidx,u_sidx_counts), key=lambda _:_[1], reverse=True):
        print("%4d : %7d  : %s " % (sid, sid_count, ""))
        tot += sid_count
    pass
    print("%4s : %7d " % ("TOT",tot))


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

    dump_boundaries(ox)
    #dump_sensorIndex(ox)

    for i, oxr in enumerate(ox):
        oxf = oxr[3].view(np.int32)

        # see okc/OpticksPhotonFlags optixrap/cu/generate.cu 
        bnd_sidx,nidx,idx,pflg  = oxf   ## nidx3 will soon become "the one" 

        nrpo = ggeo.get_triplet_index(nidx)
        nidx2,ridx,pidx,oidx = nrpo
        assert nidx2 == nidx 
        #if ridx > 0: continue   # skip photons with last intersect on instanced geometry 
        if ridx == 0: continue   # skip photons with last intersect on remainder geometry 

        bnd = np.int16(bnd_sidx >> 16)      
        sidx = np.int16(bnd_sidx & 0xffff)  

        sqh = seqhis[idx]  # photon index 
        sqm = seqmat[idx]
        msk = " %15s " % hismask.label(pflg) 
        his = "( %16x : %30s ) " % (sqh, histype.label(sqh)) 
        mat = "( %16x : %30s ) " % (sqm, mattype.label(sqm))
        print(" %5d : %6s %6s : %15s :  %s %s %s : %s " % (i, bnd, sidx, oxf[1:], msk,his,mat, nrpo) )
    pass
        
    dump_boundaries(ox)
    #dump_sensorIndex(ox)


 
