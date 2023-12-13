#!/usr/bin/env python
"""
CSGFoundry_getFrameE_Test.sh
=============================


"""

import os, numpy as np
from opticks.ana.fold import Fold


if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print("t.base:%s " % t.base) 
    print(t)
    print(repr(t))

    print("t.sframe")
    print(repr(t.sframe))

    m2w = t.m2w[0] 


