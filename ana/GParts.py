#!/usr/bin/env python

import os, logging
log = logging.getLogger(__name__)
from opticks.ana.key import keydir
from opticks.ana.prim import Solid 

class GParts(object):
    def __init__(self, kd, basedir="$TMP/GParts"):
        basedir = os.path.expandvars(basedir)
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
        return "\n".join(self.dirs)

if __name__ == '__main__':

    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)
    kd = keydir(os.environ["OPTICKS_KEY"])
    gp = GParts(kd)
    #print(gp)
 



