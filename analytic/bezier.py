#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

"""


Interactive Javascript to pick Bezier points

* http://blog.ivank.net/interpolation-with-cubic-splines.html
* https://www.desmos.com/calculator/cahqdxeshd


Cubic Bezier polynomial interpolating from P0 to P3 as u 
goes from 0 to 1 controlled by P1, P2::

    B(u) = P0*(1-u)**3 + P1*3*u*(1-u)**2 + P2*3*u**2*(1-u) + P3*u**3   

Or more in spirit of Bezier decide on begin/end points and 
control points

::

    (z1, rr1) 
    (cz1, crr1)
    (cz2, crr2)
    (z2, rr2) 


"""
import logging

log = logging.getLogger(__name__)
import pylab as plt

import numpy as np
from sympy import symbols, Poly 

def make_bezier(t,xs):
    if len(xs) == 2:
        bz = xs[0]*(1-t) + xs[1]*t
    elif len(xs) == 3:
        bz = xs[0]*(1-t)**2 + xs[1]*2*t*(1-t) + xs[2]*t**2
    elif len(xs) == 4:
        bz = xs[0]*(1-t)**3 + xs[1]*3*t*(1-t)**2 + xs[2]*3*t**2*(1-t) +  xs[3]*t**3 
    else:
        assert 0, xs
    pass
    return bz 

class Bezier(object):
    """ 
    ety is the cubic in t
    but for surface-of-revolution need cubic in z 
    where: 
        
          t = (z - z1)/(z2 - z1)

    """
    @classmethod
    def ZCoeff(cls, xy):
        bz = cls(xy) 
        return bz.zco_

    def __init__(self, xy):
        assert len(xy) == 4
        xy = np.asarray(xy, dtype=np.float32)
        x_ = xy[:,0]
        y_ = xy[:,1]

        t = symbols("t")
        x = symbols("x0,x1,x2,x3")
        y = symbols("y0,y1,y2,y3")

        etx = make_bezier(t,x)
        ety = make_bezier(t,y)

        etx_ = etx.subs([[x[0],x_[0]],[x[1],x_[1]],[x[2],x_[2]],[x[3],x_[3]]])
        ety_ = ety.subs([[y[0],y_[0]],[y[1],y_[1]],[y[2],y_[2]],[y[3],y_[3]]])

        xco = Poly(etx,t).all_coeffs()
        yco = Poly(ety,t).all_coeffs()
        xco_ = Poly(etx_,t).all_coeffs()
        yco_ = Poly(ety_,t).all_coeffs()

        log.info("xco %r %r %r  " % ( xco, xco_, x_ ))
        log.info("yco %r %r %r  " % ( yco, yco_, y_ ))

        self.x_ = x_
        self.y_ = y_
        self.t = t 
        self.etx_ = etx_
        self.ety_ = ety_

        z,z1,z2 = symbols("z,z1,z2")

        z1_ = x_[0]
        z2_ = x_[-1]

        self.z = z
        self.z1_ = z1_
        self.z2_ = z2_

        etz_ = ety_.subs(t, (z-z1)/(z2-z1) ).subs( [[z1,z1_],[z2,z2_]] )
        zco_ = Poly(etz_,z).all_coeffs()
        
        self.xco = xco
        self.yco = yco
        self.xco_ = xco_
        self.yco_ = yco_
        
        self.etz_ = etz_
        self.zco_ = zco_


    def zval(self, zz):
        return map(lambda _:self.etz_.subs(self.z,_), zz )

    def xval(self, tt):
        return map(lambda _:self.etx_.subs(self.t,_), tt )
    def yval(self, tt):
        return map(lambda _:self.ety_.subs(self.t,_), tt )

    def plot(self, ax, mx=50, my=50):

        tt = np.linspace(0,1,100)

        ax.set_xlim(bz.x_[0]-mx,bz.x_[3]+mx)
        ax.set_ylim(   0, max(bz.y_)+my)
        ax.plot(bz.xval(tt), bz.yval(tt), 'k')

        mk = "ob og og or".split() 
        for i in range(4):
            ax.plot( bz.x_[i], bz.y_[i], mk[i] )
        pass

        ax.plot((bz.x_[3],bz.x_[2]),(bz.y_[3],bz.y_[2]), 'r-')
        ax.plot((bz.x_[0],bz.x_[1]),(bz.y_[0],bz.y_[1]), 'b-')
     


if __name__ == '__main__':


    plt.ion()
    plt.close()

    xy = [[-100,30],[-50,80],[50,30],[100,100]]

    bz = Bezier(xy) 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bz.plot(ax)
    plt.show()

    print Bezier.ZCoeff(xy)

    



