#!/usr/bin/env python
"""
ht.py : quick checks on hits
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

    In [29]: ht_flags[:,1].min()
    Out[29]: 8

    In [30]: ht_flags[:,1].max()
    Out[30]: 189


::

    In [36]: ht_flags.shape
    Out[36]: (87, 4)

    In [35]: ox_flags[ox_flags[:,1] != -1].shape
    Out[35]: (431, 4) 
    ## many photons land on sensors but are not classified as hits 


::

    In [43]: ox_lander[:,1].min()
    Out[43]: 0

    In [44]: ox_lander[:,1].max()
    Out[44]: 190




"""

import os, numpy as np
from opticks.ana.histype import HisType
from opticks.ana.mattype import MatType
from opticks.ana.hismask import HisMask
from opticks.ana.blib import BLib

histype = HisType()
mattype = MatType()
hismask = HisMask() 
blib = BLib()


if __name__ == '__main__':
    ox = np.load("ox.npy")
    ht = np.load("ht.npy") # ht are a selection of the ox:photons with SD:SURFACE_DETECT flag set  

    ph = np.load("ph.npy") # seqhis, seqmat sequence histories for all photons
    seqhis = ph[:,0,0]
    seqmat = ph[:,0,1]

    ## hmm need a standard place for detector level stuff like this 
    sd = np.load(os.path.expandvars("$TMP/G4OKTest/sensorData.npy"))  
    triplet_id = sd.view(np.uint32)[:,3]  
    placement_id = ( triplet_id & 0x00ffff00 ) >> 8  

    dump = False
    if dump:
        for _ in seqhis[:10]:print("%16x : %s " %(_,histype.label(_)))
        for _ in seqmat[:10]:print("%16x : %s " %(_,mattype.label(_)))
        for htf in ht[:10,3].view(np.int32):print(htf, hismask.label(htf[3]))
    pass

    ht_flags = ht.view(np.int32)[:,3]
    ox_flags = ox.view(np.int32)[:,3]   
    ht_flags2 = ox_flags[np.where(ox_flags[:,3] & hismask.code("SD"))]  
    # duplicate hit flags from the photons, using the .w flag mask 
    # and knowledge that hit selection requires SD:SURFACE_DETECT 
    assert np.all( ht_flags == ht_flags2 )

    ht_idx = ht_flags[:,2]
    ox_lander = ox_flags[ox_flags[:,1] != -1]

    print("ox_flags : %s " % repr(ox_flags.shape) )
    print("ht_flags : %s " % repr(ht_flags.shape) )
    print("ox_lander : %s : photons landing on sensor volumes  " % repr(ox_lander.shape))  

    bndidx = np.unique(np.abs(ox_lander[:,0])) - 1    # subtract 1 to get index as signed boundaries are 1-based 
    print("bndidx : %s " % repr(bndidx))
    for _ in bndidx:print(blib.bname(_)) 


    tot = 0 
    for oxf in ox_lander:
        sidx = oxf[1] # sensor index 
        tid = triplet_id[sidx]
        pid = placement_id[sidx]
        idx = oxf[2]  # photon index 
        sqh = seqhis[idx]
        sqm = seqmat[idx]
        is_hit = idx in ht_idx
        tot += int(is_hit)
        stat = "HIT" if is_hit else "-"
        print("%20s %15s %10s    %16x : %30s    %16s : %30s  %10x %3x %3d  " % (oxf, hismask.label(oxf[3]), stat, sqh, histype.label(sqh), sqm, mattype.label(sqm), tid, pid, pid ))
    pass
    assert tot == len(ht_idx)
         
