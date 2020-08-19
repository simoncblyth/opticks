#!/usr/bin/env python
"""
::

    ipython -i ImageNPYTest.py


"""
import os 
import numpy as np
import matplotlib.pyplot as plt 
p = os.path.expandvars("$TMP/ImageNPYTest.npy")
print(p)
a = np.load(p)
print(a.shape)

plt.ion()
plt.imshow(a)





