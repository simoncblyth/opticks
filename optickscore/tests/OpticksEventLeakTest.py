#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
plt.close()


path = os.path.expandvars("$TMP/optickscore/tests/OpticksEventLeakTest.npy")
a = np.load(path)
print(a)

fig, axs = plt.subplots(1, 2)

ax = axs[0] 
ax.plot( a[:,1] )

ax = axs[1] 
ax.plot( a[:,0], a[:,1] )


plt.show()

