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
groupvel.py
============

* C++ and NumPy implementations of the ported GROUPVEL_ 
  calculation are matching 

  NB the calc interps back to original bin, 
  to avoid bin shifting sins that G4 commits


TODO:

* compare actual G4 calc against mt port of the calc
  

"""

import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main
from opticks.ana.proplib import PropLib

# values from GConstantTest 
hc = 1239.841875       # eV.nm
c_light = 299.792458   # mm/ns
wl = np.linspace(60,820,39)   # standard wavelength domain with increasing wavelenth (nm)


def g4_bintrick(aa):
    """
    :param aa:
    :return bb:

    This curious bin shifting is used by G4 GROUPVEL calc

    In order not to loose a bin
    a tricky manoever of using the 1st and last bin and 
    the average of the body bins
    which means the first bin is half width, and last is 1.5 width
    
    ::

            0  +  1  +  2  +  3  +  4  +  5        <--- 6 original values
            |    /     /     /     /      |
            |   /     /     /     /       |
            0  1     2     3     4        5        <--- still 6 
  
    """
    bb = np.zeros_like(aa)
    bb[0] = aa[0]
    bb[1:-1] = (aa[1:-1] + aa[:-2])/2.
    bb[-1] = aa[-1]
    return bb


def np_gradient(y, edge_order=1):
    """
    ::

        Take a look at np.gradient??

        The gradient is computed using second order accurate central differences
        in the interior and either first differences or second order accurate 
        one-sides (forward or backwards) differences at the boundaries. The
        returned gradient hence has the same shape as the input array.

          0    <--- (0,1)  
  
          1    <--- (0,2)/2
          2    <--- (1,3)/2
          3    <--  (2,4)/2
          4    <--  (3,5)/2
          5    <--  (4,6)/2
          6    <--  (5,7)/2
          7    <--  (6,8)/2
          8    <--  (7,9)/2

          9    <--  (8,9) 


        In [10]: y = np.arange(10, dtype='f')


        In [14]: y[:-2]   # skip bins N-2,N-1  :  ie N-2 values
        Out[14]: array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.], dtype=float32)

        In [13]: y[2:]   # skip bins 0,1 : ie N-2 values
        Out[13]: array([ 2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.], dtype=float32)

        In [15]: y[2:] - y[:-2]    # between N-2 values -> N-2 values
        Out[15]: array([ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.], dtype=float32)

        In [16]: (y[2:] - y[:-2])/2.
        Out[16]: array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.], dtype=float32)

    """
    assert edge_order == 1
    out = np.zeros_like(y)
    out[0] = y[1] - y[0]
    out[1:-1] = (y[2:]-y[:-2])/2.0
    out[-1] = y[-1] - y[-2]
    return o



def GROUPVEL_(ri_, dump=False):
    """
    Reproduce G4 GROUPVEL calc by migrating from wavelength to energy 
    domain and duplicating the tricky bin shift manoever

                c         
    vg =  ---------------        # angular freq proportional to E for light     
            n + E dn/dE

    G4 using this energy domain approach approximating the dispersion part E dn/dE as shown below

                c                  n1 - n0         n1 - n0               dn        dn    dE          
    vg =  -----------       ds = ------------  =  ------------     ~   ------  =  ---- ------- =  E dn/dE 
           nn +  ds               log(E1/E0)      log E1 - log E0      d(logE)     dE   dlogE        


    Start by convert to energy and flipping order of energy 
    and refractive values to correspond to increasing energy domain 
    """

    ri = ri_.copy()  # refractive index values on wavelength domain
    en = hc/wl[::-1]   # eV    
    ri = ri[::-1] 

    assert ri.shape == en.shape and len(ri) == len(en)

    n0,n1= ri[:-1],ri[1:]
    e0,e1 = en[:-1],en[1:]

    nn = g4_bintrick(ri)
    ee = g4_bintrick(en)

    ds = np.zeros_like(ri)  # dispersion term
    ds[1:] = (n1-n0)/np.log(e1/e0)  ## have lost a bin from the diff ... 
    ds[0] = ds[1]                   ## duping into first value

    # ds is native to midbin anyhow

    vg0 = c_light/nn
    vg = c_light/(nn + ds)

    msk = np.logical_or( vg < 0, vg > vg0)
    vgm = vg.copy()
    vgm[msk] = vg0[msk]

    vgi = np.interp( en, ee, vgm )   # linear interpolation onto original energy values


    if dump:
        labels = "hc/en,en,hc/ee,hc/ee-hc/en,ee,nn,ds,vg0,vg,msk,vgm,vgi"
        print "".join(map(lambda _:"%11s"%_, labels.split(",")))
        print np.dstack([hc/en,en,hc/ee,hc/ee-hc/en,ee,nn,ds,vg0,vg,msk,vgm,vgi])

    return vgi[::-1]




class PropTree(object):
    def __init__(self):
        self.names = os.listdir(os.path.expandvars(self.base))
    def subdir(self, sub):
        return os.path.expandvars(os.path.join(self.base, sub))
    def path(self, sub, fname):
        return os.path.expandvars(os.path.join(self.base, sub, fname + ".npy"))
    def ary(self, sub, fname):
        return np.load(self.path(sub, fname))


class replaceGROUPVEL(PropTree):
    """
    GMaterialLib::replaceGROUPVEL in debug mode writes the 
    GROUPVEL calculated from refractive index

    ::

       In [25]: rg.vg("GdDopedLS")
       Out[25]: 
       array([[  60.    ,  206.2414],
              [  80.    ,  206.2414],
              [ 100.    ,  206.2414],
              [ 120.    ,  198.1083],
              [ 140.    ,  181.5257],


    """
    base = "$TMP/replaceGROUPVEL"
    def ri_(self, matname):
        return self.ary(matname, "refractive_index")
    def vg_(self, matname):
        return self.ary(matname, "group_velocity")

    def check_all(self, dump=False):
        for name in self.names:
            self.check(name, dump=dump)

    def check(self, name="GdDopedLS", dump=True):

        rip = self.ri_(name)

        wl = rip[:,0]
        ri = rip[:,1]

        vgp = self.vg_(name)
        wl2 = vgp[:,0]
        vg = vgp[:,1]

        assert np.allclose(wl,wl2)

        vgpy = GROUPVEL_(ri)   ## python implementation of same calc 
        vgdf = vg-vgpy

        if dump:
            print np.dstack([wl,ri,vg,vgpy, vgdf])

        log.info("check %30s  %s " % (name, vgdf.max()) )  


class CGROUPVELTest(PropTree):
    """
    CMaterialLib::saveGROUPVEL which is invoked by CGROUPVELTest  
    writes arrays with the G4 calculated GROUPVEL
    """
    base="$TMP/CGROUPVELTest"
    fnam="saveGROUPVEL"

    def ri(self, matname):
        a = self.ary(matname, self.fnam)
        return a[0,::-1,2]

    def vg(self, matname):
        a = self.ary(matname, self.fnam)
        return a[1,::-1,2]



if __name__ == '__main__':
    ok = opticks_main()

    mlib = PropLib("GMaterialLib")

    name = "GdDopedLS"
    okmat = mlib(name)
    okri = okmat[0,:,0]  # refractive index values on wavelength domain, from GMaterialLib cache prior to postCache diddling 


    cg = CGROUPVELTest()
    rg = replaceGROUPVEL()




