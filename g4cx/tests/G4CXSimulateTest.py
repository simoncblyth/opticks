#!/usr/bin/env python 

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.xfold import XFold
from opticks.ana.p import * 


if __name__ == '__main__':
    a = Fold.Load(symbol="a")
    A = XFold(a, symbol="A") 

    self = A



