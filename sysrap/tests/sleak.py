#!/usr/bin/env python
"""
sleak.py
======================

::


"""

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.npmeta import NPMeta

if __name__ == '__main__':
    fold = Fold.Load("$SLEAK_FOLD", symbol="fold")
    print(repr(fold))
 

