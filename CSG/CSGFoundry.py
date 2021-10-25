#!/usr/bin/env python
import os, numpy as np, logging
log = logging.getLogger(__name__)

class CSGFoundry(object):
    FOLD = os.path.expandvars("$TMP/CSG_GGeo/CSGFoundry")
    FMT = "   %10s : %20s  : %s "
 
    def __init__(self, fold=FOLD):
        self.load(fold)

    def load(self, fold):
        log.info("load %s " % fold)

        if not os.path.isdir(fold):
            log.fatal("CSGFoundry folder %s does not exist " % fold)
            log.fatal("create foundry folder from OPTICKS_KEY geocache with CSG_GGeo/run.sh " )
            assert 0 
        pass

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

    @classmethod
    def parse_ISEL(cls, ISEL, obndnames):
        """
        :param ISEL: comma delimited list of strings or integers 
        :param obndnames: boundary names in descending frequency order
        :return isels: list of frequency order indices 

        Integers in the ISEL are interpreted as frequency order indices. 

        Strings are interpreted as fragments to look for in the ordered boundary names, 
        eg use Hama or NNVT to yield the list of frequency order indices 
        with corresponding bounary names containing those strings. 
        """
        ISELS = list(filter(None,ISEL.split(",")))

        isels = []
        for i in ISELS:
            if i.isnumeric(): 
                isels.append(int(i))
            else:
                for idesc, bn in enumerate(obndnames):
                    if i in bn:
                        isels.append(idesc)
                    pass
                pass
            pass
        pass         
        return isels 




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cf = CSGFoundry()
    cf.dump_node_boundary()


