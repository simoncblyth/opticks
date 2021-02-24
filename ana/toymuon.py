#!/usr/bin/env python

import numpy as np
import math
from scipy.spatial.transform import Rotation as R 

m = R.from_euler('z', 90, degrees=True)

try:
    import pyvista as pv
except ImportError:
    pv = None
pass

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
pass


def yan_0():

    rndmcostheta = random.uniform(-1,1);
    rndmphi = random.uniform(0,math.pi*2);
    r = math.sqrt(1-rndmcostheta*rndmcostheta);
    x = r*math.cos(rndmphi);
    y = r*math.sin(rndmphi); 

    return x,y,z 


def uniform_points_on_sphere(size=1000):

    u0 = np.random.uniform(low=-1, high=1, size=size)  
    u1 = np.random.uniform(low=0, high=np.pi*2., size=size)

    pos = np.zeros( (size,3) , dtype=np.float32) 

    azimuth = u1 
    cos_polar = u0
    sin_polar = np.sqrt( 1.-u0*u0 )

    x = sin_polar*np.cos(azimuth)
    y = sin_polar*np.sin(azimuth)
    z = cos_polar

    pos[:,0] = x 
    pos[:,1] = y 
    pos[:,2] = z 

    return pos


class Generator(object):
    def plot(self, ax):
        ax.plot(self.x,self.y) 
        ax.plot(self.x,self.cdf) 

    def __call__(self, u ):
        """
        :param u: uniform random number in range 0->1 (or numpy array of such randoms)
        :return x: generated sample that follows the origin pdf by interpolation of inverse CDF
        """
        return np.interp( u, self.idom, self.icdf )

   
class MuonZenithAngle(Generator):
    """
    Googling for "Gaisser formula plot" yields:

    "Muon Simulation at the Daya Bay Site",  Mengyun, Guan (2010)

    https://escholarship.org/uc/item/6jm8g76d

    Fig 10 from the above paper looks like a bit like   cos^2( polar-pi/4 ) - 0.5 
    so lets use that as an ad-hoc approximation. 

    For a review on "COSMIC RAY MUON PHYSICS" see 

    * https://core.ac.uk/download/pdf/25279944.pdf
    """

    def __init__(self, x, num_idom=1000):

        xp = x-0.25*np.pi 
        y = np.cos(xp)*np.cos(xp) - 0.5   
        # adhoc choice of function, should really use full Gaisser with fitted params from measuremnts   
        #y /= y.sum()
 
        cdf = np.cumsum(y)
        cdf /= cdf[-1]       # CDFs tend to 1.

        dom = x                             # domain of muon zenith angles
        idom = np.linspace(0,1, num_idom)   # domain of probability 0->1 which is domain of inverse cdf 
        icdf = np.interp( idom, cdf, dom )  # interpolated invertion y<->x of the cdf

        self.x = x        # domain of muon zenith angles
        self.y = y        # non-normalized PDF of muon zenith angles
        self.cdf = cdf
        self.icdf = icdf 
        self.idom = idom 


class MuonAzimuthAngle(Generator):
    def __init__(self, x, num_idom=1000):

        y = np.ones(len(x),dtype=np.float32)
        cdf = np.cumsum(y)
        cdf /= cdf[-1]       # CDFs tend to 1.

        dom = x                             # domain of muon zenith angles
        idom = np.linspace(0,1, num_idom)   # domain of probability 0->1 which is domain of inverse cdf 
        icdf = np.interp( idom, cdf, dom )  # interpolated invertion y<->x of the cdf

        self.x = x 
        self.y = y
        self.cdf = cdf
        self.icdf = icdf
        self.idom = idom
 


def plot3d(pos):
    pl = pv.Plotter()
    pl.add_points(pos)
    pl.show_grid()
    cp = pl.show()
    return cp

 
if __name__ == '__main__':
        
    zdom = np.linspace(0, 0.5*np.pi, 1000)  
    adom = np.linspace(0, 2.*np.pi,  1000)  

    zen = MuonZenithAngle(zdom)
    azi = MuonAzimuthAngle(adom)

    size = 10000
    u0 = np.random.uniform(size=size)
    u1 = np.random.uniform(size=size)

    zenith = zen(u0)   # generate a sample of muon zenith angles that follows the desired distribution
    azimuth = azi(u1)

    pos = np.zeros( (len(zenith),3), dtype=np.float32 )

    pos[:,0] = np.sin(zenith)*np.cos(azimuth)
    pos[:,1] = np.sin(zenith)*np.sin(azimuth)
    pos[:,2] = np.cos(zenith)

    plot3d(pos)

    fig, axs = plt.subplots(2, 2)

    zen.plot(axs[0][0])
    azi.plot(axs[0][1])

    axs[1][0].hist(zenith, bins=500)
    axs[1][1].hist(azimuth, bins=500)

    plt.show()

    #pos = uniform_points_on_sphere(size=10_000)


    

     



