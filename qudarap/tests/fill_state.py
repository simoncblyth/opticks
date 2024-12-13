#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold


if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    print(np.c_[np.arange(len(t.state)), t.state[:,4].view(np.uint32), t.state_names ]) 




    
