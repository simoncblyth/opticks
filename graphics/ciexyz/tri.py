#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

class Tri(object):
    """
    http://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    """
    def __init__(self, abc):
        A,B,C = np.array(abc)
        v0 = C - A
        v1 = B - A
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot11 = np.dot(v1, v1)
        invDenom = 1. / (dot00 * dot11 - dot01 * dot01)

        self.A = A
        self.v0 = v0
        self.v1 = v1
        self.dot00 = dot00
        self.dot01 = dot01
        self.dot11 = dot11
        self.invDenom = invDenom

    def inside(self, P):
        """
        works with grids of P
        """ 
        v2 = np.array(P) - self.A
        dot12 = np.dot(v2, self.v1)
        dot02 = np.dot(v2, self.v0)
        u = (self.dot11 * dot02 - self.dot01 * dot12) * self.invDenom
        v = (self.dot00 * dot12 - self.dot01 * dot02) * self.invDenom
        return np.logical_and(np.logical_and( u >= 0., v >= 0.), u+v < 1.) 
        #return np.logical_and( u >= 0., v >= 0.)



def triplot():
    """
    http://stackoverflow.com/questions/22121239/matplotlib-imshow-default-colour-normalisation
    http://colorspacious.readthedocs.org/en/latest/tutorial.html
    """
    nx = ny = 500
    yr = slice(0,2,complex(0,ny))
    xr = slice(0,2,complex(0,nx))
    y, x = np.mgrid[yr,xr]
    xy = np.dstack((x,y))   # eg shape 10,10,2
    
    t = Tri([[0.6,0.9],[1.5,1.2],[1.8,0.2]])
    b = t.inside(xy) 

    RGBA = np.zeros( (len(x),len(y), 4 )) 
    RGBA[:,:,0] = np.ones((nx,ny))
    RGBA[:,:,1] = np.zeros((nx,ny))
    RGBA[:,:,2] = np.ones((nx,ny))
    RGBA[:,:,3] = np.zeros((nx,ny))

    RGBA[b,3] = 1.

    plt.ion()

    fig, ax = plt.subplots()

    im = ax.imshow(RGBA, origin="lower", extent=[xr.start,xr.stop,yr.start,yr.stop], aspect=1, alpha=1, vmin=0, vmax=1)

    fig.colorbar(im)
    plt.show()


if __name__ == '__main__':
    triplot()


