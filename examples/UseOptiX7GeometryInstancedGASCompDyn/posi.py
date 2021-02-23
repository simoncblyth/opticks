#!/usr/bin/env python

import os, numpy as np
path_ = lambda:os.path.expandvars("/tmp/$USER/opticks/%(name)s/npy/%(name)s.npy") % dict(name=os.path.basename(os.getcwd()))


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
    a = np.load(path_())
    print(a)

    t = a[:,:,3]   
    w = np.where( t > 0 )      # find pixels with intersects
    b = a[w]                   # select intersect coordinates
    lpos = b[:,:3]             # skip the 4th 
    sz = 100.0                 # known radius
    d = sdf_sphere(lpos, sz)   # sdf : distances to sphere surface 

    print("d.min : ", d.min())
    print("d.max : ", d.max())

