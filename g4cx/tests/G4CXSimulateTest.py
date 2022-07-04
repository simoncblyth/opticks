#!/usr/bin/env python 

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.p import * 


if __name__ == '__main__':
    t = Fold.Load()
    print(t)


