#!/usr/bin/env python

import os, numpy as np, logging
log = logging.getLogger(__name__)
from opticks.CSG.CSGFoundry import CSGFoundry 

if __name__ == '__main__':
    cf = CSGFoundry.Load()
    print(repr(cf))
pass

