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


import numpy as np
import logging
log = logging.getLogger(__name__)
from opticks.ana.base import opticks_main
from opticks.analytic.csg import CSG  

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
#import pylab as plt


class Cubic(object):
    def __init__(self, A, B, C, D, z1, z2):
        self.A = A 
        self.B = B 
        self.C = C 
        self.D = D
        self.z1 = z1
        self.z2 = z2

    def __repr__(self):
        return "Cubic(A=%s,B=%s,C=%s,D=%s,z1=%s,z2=%s)" % (self.A, self.B, self.C, self.D, self.z1, self.z2 )
  
    def as_csg(self):
        return CSG.MakeCubic(A=self.A, B=self.B, C=self.C, D=self.D, z1=self.z1, z2=self.z2)    

    def rrmax(self):
        """
        ::

             v = A z**3 + B z**2 + C z + D 

             dv/dz =   3 A z**2 + 2 B z + C

        """
        A,B,C,D,z1,z2 = self.A, self.B, self.C, self.D, self.z1, self.z2

        d = 3*A
        disc = B*B - d*C

        vals = np.zeros( 4 )
        vals[0] = self.rrval(z1)
        vals[1] = self.rrval(z2)

        if disc > 0:
            sdisc = np.sqrt(max(disc,0.))   
            q = -(B + sdisc) if B > 0 else -(B - sdisc) 
           
            e1 = q/d 
            e2 = C/q
           
            vals[2] = self.rrval(e1) if e1 > z1 and e1 < z2 else 0. 
            vals[3] = self.rrval(e2) if e2 > z1 and e2 < z2 else 0. 
        pass
        return vals.max()


    def rrval(self, z ):
        A,B,C,D = self.A, self.B, self.C, self.D
        return (((A*z+B)*z + C)*z) + D          

    def rval(self, z ):
        return np.sqrt(self.rrval(z))


    def plot_profile(self, ax):
        z = np.linspace(self.z1,self.z2,100)

        zlen = self.z2 - self.z1 
        rrmx = self.rrmax()

        rmi = 0
        rmx = np.sqrt(rrmx)

        rlen = rmx - rmi
        rlenp = rlen+rlen/10.
 
        ax.set_ylim(self.z1-zlen/10.,self.z2+zlen/10.)
        ax.set_xlim( -rlenp , rlenp )
      
        rv = self.rval(z)
 
        ax.plot(rv, z, 'k')
        ax.plot(-rv, z, 'k')

        ax.plot((-rv[0],rv[0]),(z[0],z[0]), 'r-')
        ax.plot((-rv[-1],rv[-1]),(z[-1],z[-1]), 'r-')


   
    def plot_sor(self, ax):
        # https://stackoverflow.com/questions/36464982/ploting-solid-of-revolution-in-python-3-matplotlib-maybe
         
        A,B,C,D,z1,z2 = self.A, self.B, self.C, self.D, self.z1, self.z2

        u = np.linspace(z1,z2,100)
        v = np.linspace(0, 2*np.pi, 60)

        U, V = np.meshgrid(u, v)

        R = self.rval(U)

        X = R*np.cos(V)
        Y = R*np.sin(V)
        Z = U

        ax.plot_surface(X, Y, Z, alpha=0.3, color='red', rstride=6, cstride=12)





if __name__ == '__main__':

    args = opticks_main(csgpath="$TMP/cubic_py")

    CSG.boundary = args.testobject
    CSG.kwa = dict(poly="IM", resolution="50")

    container = CSG("box", param=[0,0,0,200], boundary=args.container, poly="MC", nx="20" )

    cubic = Cubic(A=1, B=10, C=100, D=400, z1=-10, z2=10) 
    log.info("cubic:%r " % cubic )
    a = cubic.as_csg()   

    #zrrs = [[-100,30],[-50,80],[50,30],[100,100]]
    #a = CSG.MakeCubicBezier(zrrs)

    CSG.Serialize([container, a], args.csgpath )


    plt.ion()
    plt.close()
    
    fig = plt.figure()

    ax = fig.add_subplot(121)
    cubic.plot_profile(ax)

    ax = fig.add_subplot(122, projection='3d')
    cubic.plot_sor(ax)


    plt.show()




