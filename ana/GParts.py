#!/usr/bin/env python

import os, sys, logging, argparse, numpy as np
log = logging.getLogger(__name__)
from opticks.ana.key import keydir
from opticks.ana.prim import Solid 

class GParts(object):
    """
    Opticks executables run with option --savegparts Opticks::isSaveGPartsEnabled() 
    """
    def __init__(self, kd, basedir="$TMP/GParts"):
        basedir = os.path.expandvars(basedir)

        if not os.path.isdir(basedir):
            log.fatal("missing directory %s " % basedir)
            log.fatal("create using --savegparts option with any Opticks executable that invokes GGeo::deferredCreateGParts" )
            sys.exit(1)
        pass


        ii = sorted(list(map(int,os.listdir(basedir))))
        dirs = list(map(lambda i:os.path.join(basedir, str(i)), ii)) 

        solids = []
        for d in dirs:
            log.debug(d) 
            solid = Solid(d, kd)
            solids.append(solid)
        pass

        self.dirs = dirs 
        self.solids = solids
    pass
    def __getitem__(self, i):
         return self.solids[i]

    def __repr__(self):
        return "\n".join(list(map(repr,self.solids)))



def parse_args(doc, **kwa):
    np.set_printoptions(suppress=True, precision=3, linewidth=200)
    parser = argparse.ArgumentParser(doc)
    parser.add_argument(     "ridx", nargs="*", type=int, help="GParts index to dump.")
    parser.add_argument(     "--level", default="info", help="logging level" ) 
    args = parser.parse_args()
    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
    return args  




if __name__ == '__main__':
    args = parse_args(__doc__)

    kd = keydir(os.environ["OPTICKS_KEY"])
    gp = GParts(kd)


    if len(args.ridx) > 0:
        for ridx in args.ridx:
            print(gp[ridx])
        pass
    else:
        print(gp)
    pass

 



