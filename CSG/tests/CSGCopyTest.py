#!/usr/bin/env python

import os, numpy as np
from opticks.CSG.CSGFoundry import CSGFoundry

if __name__ == '__main__':
     a = CSGFoundry.Load("$AFOLD", symbol="a")
     b = CSGFoundry.Load("$BFOLD", symbol="b")
     print(repr(a))
     print(repr(b))


