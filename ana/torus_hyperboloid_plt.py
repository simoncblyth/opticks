#!/usr/bin/env python
"""
Continuing from tboolean-12

TODO: merge this with the much better plotting technique (deferred placement) of x018_torus_hyperboloid_plt.py 

"""

import numpy as np, math 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse
import matplotlib.lines as mlines


class Tor(object):
    def __init__(self, R, r):
        self.R = R
        self.r = r

    def __repr__(self):
        return "Tor r:%s R:%s " % (self.r, self.R )

    def rz(self, z):
        R = self.R
        r = self.r
        return R - math.sqrt(r*r-z*z)  
   
class Hyp(object):
    def __init__(self, r0, zf, z1, z2):
        self.r0 = r0
        self.zf = zf
        self.z1 = z1
        self.z2 = z2

    @classmethod
    def ZF(cls, r0, zw, w ):
        """ 
        :param r0: waist radius, ie radius at z=0  
        :param zw: z at which to pin the radius
        :param w: 

        hyperboloid zf param to hit radius w, at z=zw 
        """  
        rr0 = r0*r0
        ww = w*w 
        return zw*math.sqrt(rr0/(ww-rr0)) 

    def __repr__(self):
        return "Hyp r0:%s zf:%s z1:%s z2:%s " % (self.r0, self.zf, self.z1, self.z2 ) 

    def rz(self, z):
        r0 = self.r0
        zf = self.zf
        zs = z/zf 
        return r0*np.sqrt( zs*zs + 1 )  
 
def make_rect( cxy , wh, **kwa ):
    """
    :param cxy: center of rectangle
    :param wh: width, height
    """
    ll = ( cxy[0] - wh[0]/2., cxy[1] - wh[1]/2. )
    return Rectangle( ll,  wh[0], wh[1], **kwa  ) 


if __name__ == '__main__':



    R,r,ch,cz,cn = 97.000,52.010,23.783,-23.773,-195.227

    cyr = 75.951


    r0 = R - r 
    rr0 = r0*r0

    tor = Tor(R,r)
    assert tor.rz(0) == R - r 
    assert tor.rz(r) == R  

    # in torus/hyp frame cylinder top and bottom at

    ztop, zbot = ch - cz, -ch - cz  #     (47.556, -0.010000000000001563)
    rtop, rbot = tor.rz(ztop), tor.rz(zbot)

    zf = Hyp.ZF( rbot, ztop, rtop )
    hyp = Hyp( rbot, zf, zbot, ztop )


    sz = R+1.5*r
    sz = 500 


    exy,ez = 1.391,1.000
    era = 179.00


    bulb = Ellipse( xy=(0,0), width=exy*era, height=ez*era, fill=False )  


    rhs = Circle( (R,cz),  radius=r, fill=False) 
    lhs = Circle( (-R,cz),  radius=r, fill=False) 

    cy = make_rect( (0,0), (2*cyr,2*ch), fill=False )

    byr = 45.010
    byh = 57.510
    cybase = make_rect( (0,-276.500), (2*byr, 2*byh), fill=False ) 

    cur = 254.00
    cuh = 92.000

    cycut = make_rect( (0,cuh) ,  (2*cur, 2*cuh), fill=False )


    plt.ion()
    fig = plt.figure(figsize=(5,5))
    plt.title("to_boundary")

    ax = fig.add_subplot(111)
    ax.set_ylim([-sz,sz])
    ax.set_xlim([-sz,sz])

    ax.add_patch( bulb )
    ax.add_patch( lhs )
    ax.add_patch( rhs )
    ax.add_patch( cy )
    ax.add_patch( cybase )
    ax.add_patch( cycut )

    z = np.linspace( -sz, sz, 100 )

    dz = cz

    rz = hyp.rz(z) 

    ax.plot( rz, z + dz, c="b") 

    ax.plot( -rz, z + dz, c="b") 

    
    fig.show()







