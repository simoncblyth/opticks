#!/usr/bin/env python

import os, numpy as np 
import matplotlib.pyplot as plt

a = np.load(os.path.expandvars("$NPYPATH"))
print(a.shape)

plt.imshow(a) 
plt.show()



