#!/usr/bin/env python

import os, numpy as np
import matplotlib.pyplot as plt 

np.set_printoptions(suppress=True)

dir_ = lambda:os.path.expandvars("/tmp/$USER/opticks/%(name)s/$GEOMETRY") % dict(name=os.path.basename(os.getcwd()))
path_ = lambda name:os.path.join(dir_(), name)

def sdf_sphere(p,sz):
    """ 
    :param p: intersect coordinates array of shape (n,3)
    :param sz: scalar radius of sphere
    :return d: distance to sphere surface, array of shape (n,) : -ve inside 

    https://iquilezles.org/www/articles/distfunctions/distfunctions.htm

    ::

        float sdSphere( vec3 p, float s )
        {
            return length(p)-s;
        }

    """
    assert len(p.shape) == 2 and p.shape[-1] == 3 and p.shape[0] > 0 
    d = np.sqrt(np.sum(p*p, axis=1)) - sz
    return d


if __name__ == '__main__':

    posi = path_("posi.npy")

    ias0 = path_("ias_0.npy")

    gas0 = path_("gas_0.npy")
    gas1 = path_("gas_1.npy")
    gas2 = path_("gas_2.npy")

    print(posi)
    print(ias0)
    px = np.load(posi)
    i0 = np.load(ias0)
    g0 = np.load(gas0)
    g1 = np.load(gas1)
    g2 = np.load(gas2)


    print(px)

    i = px[:,:,3].view(np.uint32)   
    w = np.where( i > 0 )         # find pixels with identity info
    b = px[w]                      # select intersect coordinates
    lpos = b[:,:3]                # skip the 4th 
    sz = 100.0                    # known radius
    d = sdf_sphere(lpos, sz)      # sdf : distances to sphere surface 

    print("d.min : ", d.min())
    print("d.max : ", d.max())


    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(i) 
    fig.show()


