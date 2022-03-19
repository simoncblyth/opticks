#!/usr/bin/env python

import os, numpy as np


TEST = os.environ["TEST"]
FOLD = os.path.expandvars("/tmp/$USER/opticks/QSimTest")

if __name__ == '__main__':
    path = os.path.join(FOLD, "%s.npy" % TEST)
    s = np.load(path)
    print(" TEST %s s %s path %s " % (TEST, str(s.shape), path))



    
