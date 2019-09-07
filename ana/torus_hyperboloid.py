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


class Tor(object):
    def __init__(self, R, r):
        self.R = R
        self.r = r

    def __repr__(self):
        return "Tor r:%s R:%s " % (self.r, self.R )

    def rz(self, _z):
        z = np.asarray(_z)
        R = self.R
        r = self.r
        return R - np.sqrt(r*r-z*z)  
  
 
class Hyp(object):
    def __init__(self, r0, zf ):
        """
        :param r0: waist radius, ie radius at z=0
        :param zf: param as returned by ZF
        """
        self.r0 = r0
        self.zf = zf

    def _stereo(self):
        """
        X4Solid::convertHype_::

            1082     /*
            1083      Opticks CSG_HYPERBOLOID uses
            1084                 x^2 +  y^2  =  r0^2 * (  (z/zf)^2  +  1 )
            1085 
            1086      G4Hype uses
            1087                 x^2 + y^2 = (z*tanphi)^2 + r^2
            1088                 x^2 + y^2 =  r0^2 * ( (z*tanphi/r0)^2 + 1 )
            1089 
            1090      So     
            1091                tanphi/r0 = 1/zf
            1092 
            1093                zf = r0/tanphi
            1094 
            1095                tanphi = r0/zf
            1096 
            1097             stereo = phi = arctan(r0/zf)
            1098 
            1099     */
        """
        return np.arctan2( self.r0, self.zf )

    stereo = property(lambda self:self._stereo())


    @classmethod
    def ZF(cls, r0, zw, rw ):
        """ 
        :param r0: waist radius, ie radius at z=0  
        :param zw: z at which to pin the radius
        :param rw: target radius at z=zw
        :return zf: hyperboloid zf param to hit target radius w, at z=zw 
        """  
        rr0 = r0*r0
        ww = rw*rw 
        return zw*np.sqrt(rr0/(ww-rr0)) 

    def __repr__(self):
        return "Hyp r0:%s zf:%s stereo(radians):%s  " % (self.r0, self.zf, self.stereo ) 

    def rz(self, _z):
        z = np.asarray(_z)
        r0 = self.r0
        zf = self.zf
        zs = z/zf 
        return r0*np.sqrt( zs*zs + 1 )  


if __name__ == '__main__':
     

    R,r = 97.000,52.010

    r0 = R - r 
    rr0 = r0*r0

    tor = Tor(R,r)
    assert tor.rz(0) == R - r 
    assert tor.rz(r) == R  
    assert np.all(tor.rz([0,r]) == np.asarray( [R-r, R] ) )

    print tor


    zw = r 
    rw = R 
    zf = Hyp.ZF( r0, zw, rw )
    hyp = Hyp( r0, zf )

    assert hyp.rz(0) == r0  
    assert hyp.rz(zw) == rw  
    
    z = np.linspace(0,zw, 100)
    print hyp.rz(z)


    print "zf", zf






 
