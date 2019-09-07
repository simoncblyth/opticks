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

Maximum global transform deviation just less than 0.1 mm when using float64 (jumping around up to 0.2mm when using float32)
when comparing global transforms from the G4DAE/GMergedMesh 
cache and the products of the glTF json parsed matrices.

TODO: perhaps more precision loss later... compare transforms at point of use inside C++

::

In [190]: run gltf.py
[2017-07-01 11:35:10,972] p18709 {/Users/blyth/opticks/ana/base.py:266} INFO - envvar OPTICKS_ANA_DEFAULTS -> defaults {'src': 'torch', 'tag': '1', 'det': 'concentric'} 
args: gltf.py
               scenes : 1 
                nodes : 12230 
               meshes : 249 
 lpo                             [0, 0, 0, 1] : gd32(    0.00000    0.06326 )  gd64(     0.00000    0.09817 ) 
 lpo                 [0.001, 0.001, 0.001, 1] : gd32(    0.00000    0.06326 )  gd64(     0.00000    0.09817 ) 
 lpo                    [1000, 1000, 1000, 1] : gd32(    0.00000    0.06326 )  gd64(     0.00000    0.09783 ) 
 lpo                 [10000, 10000, 10000, 1] : gd32(    0.00000    0.18775 )  gd64(     0.00000    0.09497 ) 
 lpo  [10000.0001, 10000.0001, 10000.0001, 1] : gd32(    0.00000    0.18775 )  gd64(     0.00000    0.09497 ) 
 lpo                 [20000, 20000, 20000, 1] : gd32(    0.00000    0.12727 )  gd64(     0.00000    0.09218 ) 
 lpo                          [1000, 0, 0, 1] : gd32(    0.00000    0.06326 )  gd64(     0.00000    0.09833 ) 
 lpo                          [0, 1000, 0, 1] : gd32(    0.00000    0.06326 )  gd64(     0.00000    0.09771 ) 
 lpo                          [0, 0, 1000, 1] : gd32(    0.00000    0.06326 )  gd64(     0.00000    0.09813 ) 
 lpo                 [-1000, -1000, -1000, 1] : gd32(    0.00000    0.12502 )  gd64(     0.00000    0.09851 ) 
 lpo                 [-5000, -5000, -5000, 1] : gd32(    0.00000    0.12514 )  gd64(     0.00000    0.09992 ) 
 lpo                    [5000, 5000, 5000, 1] : gd32(    0.00000    0.06443 )  gd64(     0.00000    0.09652 ) 
 lpo                         [10000, 0, 0, 1] : gd32(    0.00000    0.06398 )  gd64(     0.00000    0.09997 ) 


"""

import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main, json_load_ , idp_
from opticks.analytic.glm import mdotr_


class NN(list):
    def __init__(self, volpath):
        list.__init__(self,volpath) 
    def __repr__(self):
        return "\n".join(map(repr, self))

    def _get_transforms32(self):
        return map(lambda n:n.transform32, self)
    transforms32 = property(_get_transforms32)

    def _get_transforms64(self):
        return map(lambda n:n.transform64, self)
    transforms64 = property(_get_transforms64)

    gtr32 = property(lambda self:mdotr_(self.transforms32))
    gtr64 = property(lambda self:mdotr_(self.transforms64))
    gtr32r = property(lambda self:mdotr_(self.transforms32[::-1]))
    gtr64r = property(lambda self:mdotr_(self.transforms64[::-1]))


class N(object):
    def __init__(self, gltfnode):
        self.gn = gltfnode
        self.children = gltfnode.get('children', [])
        self.transform32 = np.asarray( gltfnode.get('matrix'), dtype=np.float32 ).reshape(4,4)
        self.transform64 = np.asarray( gltfnode.get('matrix'), dtype=np.float64 ).reshape(4,4)

    def __repr__(self):
        return " mesh: %d matrix:%s pvn:%s " % (self.gn['mesh'], repr(self.gn['matrix']), self.gn['extras']['pvname'])      


class GLTF(object):
    def __init__(self, path="$TMP/tgltf/tgltf-gdml--.gltf", t0=None):
         self.path = path
         self.gltf = json_load_(path)
         self.t0 = t0 
         self.nn = {}

    def get_node(self, idx):
        return self.gltf['nodes'][idx]

    def get_n(self, idx):
        return self.nn[idx]

    def traverse(self):
        def traverse_r(idx,ancestors):

            volpath = ancestors[:]

            gn = self.get_node(idx) 
            n = N(gn)
            volpath.append(n)

            self.nn[idx] = NN(volpath) 

            cc = n.children

            #print "%6d : %6d %6d " % ( idx, len(volpath), len(cc))

            for c in cc:
                traverse_r(c, volpath)
            pass
        pass
        traverse_r(0, [])


    def __str__(self):
        lkeys = filter(lambda k:type(self.gltf[k]) is list, self.gltf.keys() )
        return "\n".join([" %20s : %d " % (k, len(self.gltf[k])) for k in lkeys])



    def gtransform_delta(self, lpos=[0,0,0,1]):
        return self.gtransform_delta_(lpos, np.float32), self.gtransform_delta_(lpos, np.float64)

    def gtransform_delta_(self, lpos_, dtype=np.float64):
        """
        Check distance between a local frame position transformed with the two 
        transforms
        """
        lpos = np.asarray(lpos_, dtype=dtype)

        num = len(self.nn)
        gdelta = np.zeros(num, dtype=dtype)

        for idx in range(num):
            t0 = self.t0[idx].reshape(4,4)
            nn = self.nn[idx] 

            t1 = nn.gtr32r if dtype is np.float32 else nn.gtr64r

            gpos0 = np.dot( lpos, t0 )[:3]
            gpos1 = np.dot( lpos, t1 )[:3]
            d = gpos0 - gpos1
            gdelta[idx] = np.sqrt(np.dot(d,d))
        pass
        return gdelta

                





if __name__ == '__main__':
     args = opticks_main()

     t0 = np.load(idp_("GMergedMesh/0/transforms.npy"))
     
     g = GLTF(t0=t0)
     print g 

     g.traverse()

     lpos = [
               [0,0,0,1],
               [1e-3,1e-3,1e-3,1],
               [1000,1000,1000,1],
               [10000,10000,10000,1],
               [10000.0001,10000.0001,10000.0001,1],
               [20000,20000,20000,1],
               [1000,0,0,1],
               [0,1000,0,1],
               [0,0,1000,1],
               [-1000,-1000,-1000,1],
               [-5000,-5000,-5000,1],
               [5000,5000,5000,1],
               [10000,0,0,1]
            ]


     for lpo in lpos:
         gd32,gd64 = g.gtransform_delta(lpo)
         print " lpo %40s : gd32( %10.5f %10.5f )  gd64(  %10.5f %10.5f ) " % ( repr(lpo),gd32.min(),gd32.max(), gd64.min(),gd64.max() )
     pass


     nn = g.nn[3159]


