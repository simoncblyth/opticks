#!/usr/bin/env python
"""
ncensus.py : Event Census wih array shape dumping 
=====================================================

Example output:

.. code-block:: py

    In [12]: run ncensus.py
    INFO:__main__:dump .npy with abbrev rs 
    INFO:__main__: path           /usr/local/env/opticks/BoxInBox/rstorch/--save.npy shape (500000, 10, 1, 4) 
    INFO:__main__: path               /usr/local/env/opticks/BoxInBox/rstorch/-1.npy shape (500000, 10, 1, 4) 
    INFO:__main__: path                /usr/local/env/opticks/BoxInBox/rstorch/1.npy shape (5000000, 1, 4) 
    INFO:__main__: path              /usr/local/env/opticks/dayabay/rscerenkov/1.npy shape (6128410, 1, 4) 
    INFO:__main__: path                 /usr/local/env/opticks/dayabay/rstorch/1.npy shape (500000, 10, 1, 4) 
    INFO:__main__: path                    /usr/local/env/opticks/DPIB/rstorch/4.npy shape (5000000, 1, 4) 
    INFO:__main__: path                  /usr/local/env/opticks/G4Gun/rsg4gun/-1.npy shape (0, 10, 1, 4) 
    INFO:__main__: path                 /usr/local/env/opticks/juno/rscerenkov/1.npy shape (9812580, 1, 4) 
    INFO:__main__: path          /usr/local/env/opticks/juno_backup/rscerenkov/1.npy shape (9812580, 1, 4) 
    INFO:__main__: path               /usr/local/env/opticks/PmtInBox/rstorch/-1.npy shape (500000, 10, 1, 4) 
    INFO:__main__: path               /usr/local/env/opticks/PmtInBox/rstorch/-2.npy shape (500000, 10, 1, 4) 
    INFO:__main__: path               /usr/local/env/opticks/PmtInBox/rstorch/-4.npy shape (500000, 10, 1, 4) 
    INFO:__main__: path               /usr/local/env/opticks/PmtInBox/rstorch/-5.npy shape (500000, 10, 1, 4) 
    INFO:__main__: path               /usr/local/env/opticks/PmtInBox/rstorch/-6.npy shape (500000, 10, 1, 4) 
    INFO:__main__: path                /usr/local/env/opticks/PmtInBox/rstorch/1.npy shape (5000000, 1, 4) 
    INFO:__main__: path                /usr/local/env/opticks/PmtInBox/rstorch/2.npy shape (5000000, 1, 4) 
    INFO:__main__: path                /usr/local/env/opticks/PmtInBox/rstorch/4.npy shape (5000000, 1, 4) 
    INFO:__main__: path                /usr/local/env/opticks/PmtInBox/rstorch/5.npy shape (5000000, 1, 4) 
    INFO:__main__: path                /usr/local/env/opticks/PmtInBox/rstorch/6.npy shape (5000000, 1, 4) 
    INFO:__main__: path                /usr/local/env/opticks/rainbow/rstorch/-5.npy shape (1000000, 10, 1, 4) 
    INFO:__main__: path                /usr/local/env/opticks/rainbow/rstorch/-6.npy shape (1000000, 10, 1, 4) 
    INFO:__main__: path                 /usr/local/env/opticks/rainbow/rstorch/5.npy shape (10000000, 1, 4) 
    INFO:__main__: path                 /usr/local/env/opticks/rainbow/rstorch/6.npy shape (10000000, 1, 4) 
    INFO:__main__: path                /usr/local/env/opticks/reflect/rstorch/-1.npy shape (10000, 10, 1, 4) 
    INFO:__main__: path                 /usr/local/env/opticks/reflect/rstorch/1.npy shape (100000, 1, 4) 
    INFO:__main__:dump .npy with abbrev ps 
    INFO:__main__: path           /usr/local/env/opticks/BoxInBox/pstorch/--save.npy shape (500000, 1, 4) 
    INFO:__main__: path               /usr/local/env/opticks/BoxInBox/pstorch/-1.npy shape (500000, 1, 4) 
    INFO:__main__: path                /usr/local/env/opticks/BoxInBox/pstorch/1.npy shape (500000, 1, 4) 
    INFO:__main__: path              /usr/local/env/opticks/dayabay/pscerenkov/1.npy shape (612841, 1, 4) 
    INFO:__main__: path                 /usr/local/env/opticks/dayabay/pstorch/1.npy shape (500000, 1, 4) 
    INFO:__main__: path                    /usr/local/env/opticks/DPIB/pstorch/4.npy shape (500000, 1, 4) 
    ...

"""

import os, logging
log = logging.getLogger(__name__)
import numpy as np

from opticks.ana.base import opticks_environment

class Census(object):
    def __init__(self, base):
        self.base = os.path.expandvars(base)

    def dump(self, abbrev="rs"):
        self.abbrev = abbrev
        log.info("dump .npy with abbrev %s " % abbrev )
        self.recurse(self.base, 0 )

    def recurse(self, bdir, depth):
        """
        
        """
        for name in os.listdir(bdir):
            path = os.path.join(bdir, name)
            if os.path.islink(path):
                log.info("skip link %s " % path) 
            elif os.path.isdir(path):
                self.recurse(path, depth+1)  
            else:
                root, fext = os.path.splitext(name) 
                if fext == ".npy":
                    self.visit_npy(path, depth)
                pass
            pass
        pass

    def visit_npy(self, path, depth):
        elems = path.split("/")
        sub = elems[-2]
        ab = sub[:2]
        log.debug(" path %30s depth %d sub %s ab %s" % (path, depth, sub, ab))
        if ab == self.abbrev:
            ary = np.load(path) 
            log.info(" path %60s shape %s " % (path, repr(ary.shape)))
        pass



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    opticks_environment() 

    c = Census("$OPTICKS_EVENT_BASE/source/evt")

    c.dump("rs")
    c.dump("ps")



        
