#!/usr/bin/env python
import os, numpy as np
import matplotlib.pyplot as plt
if not 'TMP' in os.environ:
    os.environ['TMP'] = os.path.join("/tmp",os.environ["USER"],"opticks")
pass

name = os.path.basename(os.getcwd()) 
fold = os.path.expandvars("$TMP/%s" % name)

i = np.load(os.path.join(fold,"inp.npy")) 
o = np.load(os.path.join(fold,"out.npy")) 

print(name)
print("i %s " % str(i.shape))
print("o %s " % str(o.shape))

fig, axs = plt.subplots(2)
fig.suptitle('%s' % fold)
axs[0].imshow(i)
axs[1].imshow(o)

plt.ion()
plt.show()                     

