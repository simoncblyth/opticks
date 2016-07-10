#!/usr/bin/env python

import os, numpy as np
os.environ["TMP"] = os.path.expandvars("/tmp/$USER/opticks") 

a = np.load(os.path.expandvars("$TMP/OOMinimalTest.npy"))

print a
