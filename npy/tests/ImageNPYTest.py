#!/usr/bin/env python
"""
::

    ipython -i ImageNPYTest.py


"""
import os 
import numpy as np
import matplotlib.pyplot as plt 



if 0:
    p0 = os.path.expandvars("$TMP/ImageNPYTest.npy")
    print(p0)
    a = np.load(p0)
    print(a.shape)

    plt.ion()
    plt.imshow(a)


if 1:
    p1 = "/tmp/SPPMTest_concat.npy"
    b = np.load(p1)
    print(b.shape)
    assert b.shape[0] < 4       
    layers = b.shape[0]
    fig, axs = plt.subplots(layers)
    fig.suptitle("%s %r" % (p1,b.shape))
    for i in range(layers): 
        axs[i].imshow(b[i], origin='lower')
    pass
    plt.show()                           






