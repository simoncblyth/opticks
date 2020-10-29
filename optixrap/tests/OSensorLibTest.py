#!/usr/bin/env python
"""
::
 
   ipython -i  OSensorLibTest.py

"""
import os, numpy as np

if __name__ == '__main__':

    np.set_printoptions(linewidth=200, edgeitems=10)

    a_path = os.path.expandvars("/tmp/$USER/opticks/opticksgeo/tests/MockSensorLibTest/angularEfficiency.npy")
    b_path = os.path.expandvars("/tmp/$USER/opticks/optixrap/tests/OSensorLibTest/out.npy")

    a = np.load(a_path)
    print(a_path)
    print("a:{a.shape!r}".format(a=a)) 
    assert a.shape[-1] == 1 
    a = a.reshape(a.shape[:-1])
    print("a:{a.shape!r}".format(a=a)) 
 
    b = np.load(b_path)
    print(b_path)
    print("b:{b.shape!r}".format(b=b)) 
    assert b.shape[-1] == 1 
    b = b.reshape(b.shape[:-1])
    print("b:{b.shape!r}".format(b=b)) 

    
    ctx = dict(name=os.path.basename(a_path),b=b,num_theta=b.shape[1],num_phi=b.shape[2])
    title = "{name} {b.shape!r} num_theta:{num_theta} num_phi:{num_phi}".format(**ctx)
    print(title)

    try:
        import matplotlib.pyplot as plt 
    except ImportError:
        plt = None
    pass

    if plt:
        fig, axs = plt.subplots(2)
        fig.suptitle(title)

        axs[0].imshow(a[0])
        axs[1].imshow(b[0])

        plt.ion()
        plt.show()    
    pass

    assert np.allclose( a, b )
    assert np.abs( a - b ).max() == 0.  



