#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
    f = Fold.Load("$GPropertyMap_BASE", "LS", symbol="f", order="ascend" )
    print(repr(f))

