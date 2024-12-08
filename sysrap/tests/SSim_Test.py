#!/usr/bin/env python
"""
SSim_Test.py
==============

~/o/sysrap/tests/SSim_Test.sh


"""
import numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))
pass


