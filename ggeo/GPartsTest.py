#!/bin/bash -l 

from opticks.ana.fold import Fold
from opticks.ana.key import keydir
from opticks.CSG.CSGFoundry import CSGFoundry 


class GParts(object):
    def __init__(self, base, ridx):
        fold = Fold.Load(base, str(ridx))

        prim = fold.primBuffer
        part = fold.partBuffer
        idx = fold.idxBuffer
        tran = fold.tranBuffer if hasattr(fold, 'tranBuffer') else []

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
            assert len(tran) == part_transform_max
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
        return "ridx %d prim %d part %d tran %d " % (self.ridx, len(self.prim), len(self.part), len(self.tran)) 



if __name__ == '__main__':

    kd = keydir(os.environ["OPTICKS_KEY"])
    cf = CSGFoundry(os.path.join(kd, "CSG_GGeo/CSGFoundry"))

    base = "$TMP/GParts"
    ridxs = sorted(os.listdir(os.path.expandvars(base)))
    g = {} 
    for ridx in ridxs:
        try:
            int(ridx)
            g[int(ridx)] = GParts(base, int(ridx)) 
        except ValueError:
            pass
        pass
    pass



