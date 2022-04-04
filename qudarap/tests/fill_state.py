#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold

FOLD = os.path.expandvars("/tmp/$USER/opticks/QSimTest/$TEST")

if __name__ == '__main__':
    t = Fold.Load(FOLD)

    print(np.c_[np.arange(len(t.state)), t.state[:,4].view(np.uint32), t.state_names ]) 




    
