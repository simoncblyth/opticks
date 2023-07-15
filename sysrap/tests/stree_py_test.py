#!/usr/bin/env python

import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.fold import Fold
from opticks.sysrap.stree import stree

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    BASE = os.path.expandvars("$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/stree")
    f = Fold.Load(BASE, symbol="f")
    print(repr(f))

    st = stree(f, symbol="st")
    print(repr(st))

    print("f:fold f.base %s   st:stree " % f.base )

