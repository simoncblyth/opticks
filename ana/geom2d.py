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
"""
import os, json, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.key import keydir
from opticks.ana.prim import Solid

class Geom2d(object):
    """
    Scratch geometry, for designing flight paths

    Note that the GParts are not standardly saved, need to run 
    an Opticks executable such as OpSnapTest with option --savegparts::

       OpSnapTest --savegparts 

    """
    def __init__(self, kd, ridx=0):
        fp = open(os.path.join(kd, "cachemeta.json"), "r")
        meta = json.load(fp)
        gcv = meta["GEOCACHE_CODE_VERSION"]
        log.info("GEOCACHE_CODE_VERSION:%s" % gcv)

        self.ridx = str(ridx)
        self.kd = kd
        self.meta = meta

        self._pvn = None
        self._lvn = None
        self.ce = np.load(os.path.join(self.kd, "GNodeLib", "all_volume_center_extent.npy"))

        dir_ = os.path.expandvars(os.path.join("$TMP/GParts",self.ridx))
        self.d = Solid(dir_, kd)     ## mm0 analytic
        self.select_gprim()
    
    def _get_pvn(self):
        if self._pvn is None:
            self._pvn =  np.loadtxt(os.path.join(self.kd, "GNodeLib/all_volume_PVNames.txt" ), dtype="|S100" )
        return self._pvn
    pvn = property(_get_pvn)
    
    def _get_lvn(self):
        if self._lvn is None:
            self._lvn =  np.loadtxt(os.path.join(self.kd, "GNodeLib/all_volume_LVNames.txt" ), dtype="|S100" )
        return self._lvn
    lvn = property(_get_lvn)


    def pvfind(self, pvname_start, encoding='utf-8'):
        """
        :param pvname_start: string start of PV name
        :return indices: array of matching indices in pvname array  

        Examples::

            In [1]: mm0.pvfind("pTarget")
            Out[1]: array([67843])

        """
        return np.flatnonzero(np.char.startswith(self.pvn, pvname_start.encode(encoding)))  


    def select_gprim(self, names=False):
        pp = self.d.prims
        sli = slice(0,None)
        gprim = []
        for p in pp[sli]:
            if p.lvName.startswith("sPlane"): continue
            if p.lvName.startswith("sStrut"): continue
            if p.lvName.startswith("sWall"): continue
            if p.numParts > 1: continue     # skip compounds
            gprim.append(p)
            log.info(repr(p)) 
            vol = p.idx[0]     # global volume index
            log.info(self.ce[vol])
            #print(str(p)) 
            if names:
                pvn = self.pvn[vol]
                lvn = self.lvn[vol]
                log.info(pvn)
                log.info(lvm)
            pass
        pass
        self.gprim = gprim 


    def dump(self):
        for i,p in enumerate(self.gprim):
            assert len(p.parts) == 1 
            pt = p.parts[0]
            print(repr(p)) 
            #print(str(p))
            #print(pt) 
            #print(pt.tran) 
 
    def render(self, ax, art3d=None):   
        sc = 1000
        for i,p in enumerate(self.gprim):
            assert len(p.parts) == 1 
            pt = p.parts[0]
            sh = pt.as_shape("prim%s" % i, sc=sc ) 
            if sh is None: 
               print(str(p))
               continue
            #print(sh)
            for pa in sh.patches():
                ax.add_patch(pa)
                if not art3d is None:
                    art3d.pathpatch_2d_to_3d(pa, z=0, zdir="y")
                pass
            pass
        pass





if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    kd = keydir()
    log.info(kd)
    assert os.path.exists(kd), kd 

    os.environ["IDPATH"] = kd    ## TODO: avoid having to do this, due to prim internals

    mm0 = Geom2d(kd, ridx=0)

    import matplotlib.pyplot as plt 
    from mpl_toolkits.mplot3d import Axes3D 
    import mpl_toolkits.mplot3d.art3d as art3d

    plt.ion()
    fig = plt.figure(figsize=(6,5.5))
    ax = fig.add_subplot(111,projection='3d')
    plt.title("mm0 geom2d")
    sz = 25

    ax.set_xlim([-sz,sz])
    ax.set_ylim([-sz,sz])
    ax.set_zlim([-sz,sz])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


    mm0.render(ax, art3d=art3d)






