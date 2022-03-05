#!/bin/bash -l 

import logging
log = logging.getLogger(__name__)

from opticks.ana.fold import Fold
from opticks.ana.key import keydir
from opticks.CSG.CSGFoundry import CSGFoundry 


class GParts(object):
    """

    prim 
        partOffset, numParts, tranOffset, planOffset

    part
         



        In [9]: g[0].idx.view(np.int32)                                                                                                                                                                       
        Out[9]: 
        array([[     0,    138,    138,     -1],
               [     1,     17,     17,     -1],
               [     2,      2,      2,     -1],
               [     3,      1,      1,     -1],
               [     4,      0,      0,     -1],
               ...,
               [ 70964,    103,    103,     -1],
               [322249,    126,    126,     -1],
               [322250,    123,    123,     -1],
               [322251,    124,    124,     -1],
               [322252,    125,    125,     -1]], dtype=int32)


    In [6]: g[0].prim[np.where( g[0].idx[:,1] == 103 )]                                                                                                                                                   
    Out[6]: 
    array([[16087,   127,  6639,     0],
           [16214,   127,  6647,     0],
           [16341,   127,  6655,     0],
           [16468,   127,  6663,     0],
           [16595,   127,  6671,     0],
           [16722,   127,  6679,     0],
           [16849,   127,  6687,     0],
           [16976,   127,  6695,     0],



    """
    def __init__(self, base, ridx):
        fold = Fold.Load(base, str(ridx))

        prim = fold.primBuffer
        part = fold.partBuffer
        idx = fold.idxBuffer
        tran = fold.tranBuffer if hasattr(fold, 'tranBuffer') else []

        tran_num = len(tran)

        prim_partOffset = prim[:,0]
        prim_numParts   = prim[:,1]
        prim_tranOffset = prim[:,2]
        prim_planOffset = prim[:,3]

        prim_numParts_sum = prim_numParts.sum()  
        assert prim_numParts_sum == len(part)

        part_typecode = part.view(np.int32)[:,2,3]
        part_transform = part.view(np.int32)[:,3,3] & 0x7ffffff  
        part_complement = part.view(np.int32)[:,3,3] & 0x80000000 != 0
        part_transform_max = part_transform.max()  
        
        if len(tran) > 0:
            expect_equal = tran_num == part_transform_max
            log.info(" tran_num %d part_transform_max %d expect_equal %d  " % (tran_num, part_transform_max, expect_equal ))
            assert expect_equal, " maybe did not run with option --gparts_transform_offset "  
        pass

        self.prim = prim
        self.prim_partOffset = prim_partOffset
        self.prim_numParts   = prim_numParts 
        self.prim_tranOffset = prim_tranOffset
        self.prim_planOffset = prim_planOffset
        self.prim_numParts_sum   = prim_numParts_sum 

        self.part = part
        self.part_typecode = part_typecode
        self.part_transform =  part_transform
        self.part_complement = part_complement 
        self.part_transform_max =  part_transform_max

        self.base = base
        self.ridx = ridx
        self.fold = fold

        self.tran = tran
        self.idx = idx

    def __repr__(self):
        return "ridx %d idx %s prim %s part %s tran %d " % (self.ridx, str(self.idx.shape), str(self.prim.shape), str(self.part.shape), len(self.tran)) 



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    arg = os.environ.get("OPTICKS_GEOCACHE_HOOKUP_ARG", None)
    kd = keydir(os.environ["OPTICKS_KEY"])
    cfdir = os.path.join(kd, "CSG_GGeo/CSGFoundry")
    gpdir = os.path.join(kd, "DebugGParts")

    all_ridxs = map(int,sorted(os.listdir(os.path.expandvars(gpdir)))) if os.path.isdir(gpdir) else []
    ridxs = (0,) 

    log.info("arg   : %s " % arg ) 
    log.info("kd    : %s " % kd ) 
    log.info("cfdir : %s " % cfdir) 
    log.info("gpdir : %s " % gpdir) 
    log.info("all_ridxs : %s " % str(all_ridxs) ) 
    log.info("ridxs : %s " % str(ridxs) ) 
    log.info("g[0] ...")

    cf = CSGFoundry(cfdir) if os.path.isdir(cfdir) else None

    g = {} 
    if os.path.isdir(gpdir):
        for ridx in ridxs:
            g[ridx] = GParts(gpdir, ridx)  
        pass
    pass

    ridx_midx_prim_ = lambda ridx,midx:g[ridx].prim[np.where( g[ridx].idx[:,1] == midx )]  

