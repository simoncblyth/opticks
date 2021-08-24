#!/usr/bin/env python
import os, numpy as np, logging
log = logging.getLogger(__name__)

class CSGFoundry(object):
    FOLD = os.path.expandvars("$TMP/CSG_GGeo/CSGFoundry")
    FMT = "   %10s : %20s  : %s "
 
    def __init__(self, fold=FOLD):
        self.load(fold)

    def load(self, fold):
        logging.info("load %s " % fold)
        names = os.listdir(fold)
        for name in filter(lambda name:name.endswith(".npy") or name.endswith(".txt"), names):
            path = os.path.join(fold, name)
            stem = name[:-4]
            a = np.load(path) if name.endswith(".npy") else np.loadtxt(path, dtype=np.object)
            if name == "bnd.txt": stem = "bndname"  ## TODO: avoid clash of stems between bnd.npy and bnd.txt ?
            setattr(self, stem, a)
            globals()[stem] = a 
            print(self.FMT % (stem, str(a.shape), path))
        pass

    def dump_node_boundary(self):
        logging.info("dump_node_boundary") 
        node = self.node
        bndname = self.bndname

        node_boundary = node.view(np.uint32)[:,1,2]
        ubs, ubs_count = np.unique(node_boundary, return_counts=True)

        for ub, ub_count in zip(ubs, ubs_count):
            print(" %4d : %6d : %s " % (ub, ub_count, bndname[ub]))
        pass 


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cf = CSGFoundry()
    cf.dump_node_boundary()


