#!/usr/bin/env python

import os, logging, numpy as np
log = logging.getLogger(__name__)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from env.numerics.npy.ana import Evt, Selection, costheta_, cross_
from env.numerics.npy.geometry import Boundary   
from env.numerics.npy.cie  import CIE
deg = np.pi/180.



class SphereReflect(object):
    def __init__(self, evt):
        sel = Selection(evt, "BR SA") 
    
        p0 = sel.recpost(0)[:,:3]
        p1 = sel.recpost(1)[:,:3]
        p_in = p1 - p0  

        pp = sel.recpost(1)[:,:3]
        pl = sel.recpost(2)[:,:3]
        p_out = pl - pp

        self.p0 = p0
        self.p1 = p1
        self.pl = pl
        self.p_in = p_in
        self.p_out = p_out

    def check_radius(self):
        """
        Asymmetric wider distribution than expected, from ~99 to 101.4 until
        fixed sphere positioning, when get expected symmetry

        ::

            In [50]: r.min()
            Out[50]: 99.96880356309731

            In [51]: r.max()
            Out[51]: 100.03083354506256

            In [53]: np.average(r)
            Out[53]: 100.00002525999686


        """
        r = np.linalg.norm(self.p1, 2, 1)
        log.info("r min/max %s %s " % (r.min(), r,max() )) 

        plt.hist(r, bins=100)

        return r 

    def intersection(self):
        """
        """
        origin = np.array([0,0,0])
        radius = 100.

        direction = np.array([1,0,0])

        ray_origin = self.p0
        ray_direction = np.tile(direction, len(ray_origin)).reshape(-1,3)
        center = np.tile(origin, len(ray_origin)).reshape(-1,3)

        O = ray_origin - center 
        D = ray_direction

        b = np.sum( O*D, axis=1 )
        c = np.sum( O*O, axis=1 ) - radius*radius

        disc = b*b-c    # hmm potential unhealthy subtraction of two large values

        #assert np.all(disc > 0)   

        msk = disc > 0 

        sdisc = np.sqrt(disc)
        root1 = -b -sdisc

        p1c = root1[:,None]*ray_direction + ray_origin

        nrm = (O + D*root1[:,None])/radius

        return p1c, nrm, msk 


    def check_surface_normal(self):
        """

        In [187]: nnrm = np.linalg.norm(nrm, 2, 1)

        In [188]: nnrm[sr.msk].min()
        Out[188]: 0.99999999999999667

        In [189]: nnrm[sr.msk].max()
        Out[189]: 1.0000000000000033

        """
        sr = self
        p1c, nrm, msk = sr.intersection()
        nnrm = np.linalg.norm(nrm, 2, 1)
 

    def check_intersection(self):
        """

            In [155]: sr.mdp1[:,0].min()
            Out[155]: -1.7650434697386426

            In [156]: sr.mdp1[:,0].max()
            Out[156]: 1.8097476126588909

        """
        sr = self

        p1c, nrm, msk = sr.intersection()
        p1 = self.p1

        rperp = np.sqrt(np.sum( p1[:,1:3]*p1[:,1:3] , axis=1))
        dp1 = p1c - p1
        mdp1 = dp1[msk]

        self.p1c = p1c
        self.msk = msk
        self.dp1 = dp1
        self.mdp1 = mdp1
        self.rperp = rperp


        #sr = self
        #plt.hist2d(sr.rperp[sr.msk], sr.mdp1[:,0], bins=100)   # largest deviations are tangential



    def spatial(self):
        """
        Initial observation of asymmetry,  

        *FIXED: THE SPHERE WAS OFFSET (-1,1): A HANGOVER TO AVOID LEAKY TRIANGLE CRACKS*
        after placing sphere at origin no asymmetry apparent

        ::

            In [44]: r.min()
            Out[44]: 99.96880356309731

            In [45]: r.max()
            Out[45]: 100.03083354506256

 
        """
        fig = plt.figure()

        x0 = self.p0[:,0]
        y0 = self.p0[:,1]
        z0 = self.p0[:,2]

        x1 = self.p1[:,0]
        y1 = self.p1[:,1]
        z1 = self.p1[:,2]


        nr = 2
        nc = 3 
        nb = 100 

        ax = fig.add_subplot(nr,nc,1)
        plt.hist2d(x0, y0, bins=nb) 
        ax.set_xlabel("x0 y0")  

        ax = fig.add_subplot(nr,nc,2)
        plt.hist2d(x0, z0, bins=nb)  
        ax.set_xlabel("x0 z0")  

        ax = fig.add_subplot(nr,nc,3)
        plt.hist2d(y0, z0, bins=nb)  
        ax.set_xlabel("y0 z0")  


        ax = fig.add_subplot(nr,nc,4)
        plt.hist2d(x1, y1, bins=nb)  # xy: not symmetric, seems -Y tangentials favored over +Y tangentials 
        ax.set_xlabel("x1 y1")  

        ax = fig.add_subplot(nr,nc,5)
        plt.hist2d(x1, z1, bins=nb)  # xz: only 0:-100 as only half illuminated
        ax.set_xlabel("x1 z1")  

        ax = fig.add_subplot(nr,nc,6)
        plt.hist2d(y1, z1, bins=nb)  # yz: looks symmetric
        ax.set_xlabel("y1 z1")  




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    plt.ion()
    plt.close()

    boundary = Boundary("Vacuum///MainH2OHale")



    evt = Evt(tag="1", det="rainbow")

    sr = SphereReflect(evt)

    p1 = sr.p1

    #sr.spatial()

    sr.check_intersection()




