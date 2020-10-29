#!/usr/bin/env python
"""
::

   ipython -i  OCtx3dTest.py

"""
import os, numpy as np

tmpl = "/tmp/$USER/opticks/optixrap/tests/OCtx3dTest/test_populate_3d_buffer/arr_%d.npy"
path0 = os.path.expandvars(tmpl % 0)
path1 = os.path.expandvars(tmpl % 1)
a0 = np.load(path0)
a1 = np.load(path1)






