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
Hmm need to make connection to the volume traversal index 
"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.key import keydir
from opticks.ana.prim import Dir
from opticks.ana.geom2d import Geom2d
from opticks.ana.flight import Flight

np.set_printoptions(suppress=True)

dtype = np.float32


try:
    import matplotlib.pyplot as plt 
    from mpl_toolkits.mplot3d import Axes3D 
    import mpl_toolkits.mplot3d.art3d as art3d
except ImportError:
    plt = None
pass


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    ok = os.environ["OPTICKS_KEY"]
    kd = keydir(ok)
    log.info(kd)
    assert os.path.exists(kd), kd 

    os.environ["IDPATH"] = kd    ## TODO: avoid having to do this, due to prim internals

    mm0 = Geom2d(kd, ridx=0)     ## TODO: renamings/splittings more suited to new all-in-GNodeLib-not-mm0 geometry model 

    target = mm0.pvfind('pTarget')    ## pick a convenient frame of operation 
    assert len(target) == 1 
    target = target[0]

    sc = mm0.ce[target][3]/1000.      ##  big radius in meters 17.760008

    if plt:
        plt.ion()
        fig = plt.figure(figsize=(9,9))
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
    pass

    ####################
    n9 = 7
    ezz = np.zeros( (n9,3), dtype=dtype )
    lzz = np.zeros( (n9,3), dtype=dtype )
    uzz = np.zeros( (n9,3), dtype=dtype )
    czz = np.zeros( (n9,4), dtype=dtype )

    ezz[:,0] = [-3,-2,-1,0,-1,-2,-3] 
    ezz[:,1] = np.zeros( n9, dtype=dtype ) 
    ezz[:,2] = np.zeros( n9, dtype=dtype )
    lzz[:n9] = [1,0,0]
    uzz[:n9] = [0,0,1] 
    czz[0].view("|S2")[0:8] = ["AA", "C0", "T5", "B2", "Q1", "", "N2", "" ]
    czz[2].view("|S2")[0:8] = ["","","","", "A1","","","" ] 
    czz[n9//2].view("|S2")[0:8] = ["T6", "", "", "", "", "", "", "" ]
    # AA: home the animation as ascii A - 0 is greater than the number of modes 
    # Q1:invis GlobalStyle


    fzz = Flight(ezz,lzz,uzz,czz) 

    ####################
    n0 = 3
    eaa = np.zeros( (n0,3), dtype=dtype )
    laa = np.zeros( (n0,3), dtype=dtype )
    uaa = np.zeros( (n0,3), dtype=dtype )
    caa = np.zeros( (n0,4), dtype=dtype )

    eaa[:,0] = np.linspace(-n0, -1, n0, dtype=dtype ) 
    eaa[:,1] = np.zeros( n0, dtype=dtype ) 
    eaa[:,2] = np.ones( n0, dtype=dtype )

    uaa[:n0] = [0,0,1] 

    caa[0].view("|S2")[0:8] = ["C1", "T6", "B2", "Q0", "A1", "P1", "N1", "" ]
    caa[2].view("|S2")[0:8] = ["Q1","","","Q0","C0","T5","",""] 

    laa[:-1] = eaa[1:]
    laa[-1] = [0,0,1]

    faa = Flight(eaa,laa,uaa,caa) 

    #################

    pz = 1.0
    pr = 1.05

    phase0 = np.arccos(pz) 
    ta = np.linspace( 0, 2*np.pi, 32 )[:-1]
    za = np.cos(ta+phase0)
    m = np.argmin(np.abs(za[2:]-pz))+2   # index of za closest to that pz value going around again, excluding 0
    t0 = ta[:m+1]
    n1 = len(t0)
    st0 = np.sin(t0+phase0)
    ct0 = np.cos(t0+phase0)

    ebb = np.zeros( (n1,3) , dtype=dtype )  # eye position 
    lbb = np.zeros( (n1,3) , dtype=dtype ) # up direction
    ubb = np.zeros( (n1,3) , dtype=dtype ) # up direction
    cbb = np.zeros( (n1,4), dtype=dtype )  # ctrl  

    ebb[:,0] = st0
    ebb[:,1] = 0
    ebb[:,2] = ct0
    ebb *= pr

    ubb[:,0] = st0
    ubb[:,1] = 0 
    ubb[:,2] = ct0 

    cbb[0].view("|S2")[0:8] = ["C0", "","","","T6","","T7","" ] 
    cbb[1].view("|S2")[0:1] = ["T8"] 
    cbb[n1//2+0].view("|S2")[0:8] = ["E3","","","","","","","" ]
    cbb[n1//2+1].view("|S2")[0:8] = ["","","","","","","","" ]
    cbb[n1//2+2].view("|S2")[0:8] = ["","","","","","","","" ]
    cbb[n1//2+3].view("|S2")[0:8] = ["","","","","","","","" ]
    cbb[n1//2+4].view("|S2")[0:8] = ["E1","","","","","","","" ]

    cbb[n1-2].view("|S2")[0:1] = ["T5"] 

    lbb[:-1] = ebb[1:]
    lbb[-1] = [0,0,1]

    fbb = Flight(ebb,lbb,ubb,cbb) 

    ##############################

    # take the last point x value (close to pz) and make xy loop
    r2 = np.abs(ebb[-1,0])
    tb = np.linspace( 0, 2*np.pi, 16)[:-1]
    n2 = len(tb)

    ecc = np.zeros( (n2,3), dtype=dtype )
    lcc = np.zeros( (n2,3), dtype=dtype )
    ucc = np.zeros( (n2,3), dtype=dtype )
    ccc = np.zeros( (n2,4), dtype=dtype )

    ecc[:,0] = r2*np.cos(tb)
    ecc[:,1] = r2*np.sin(tb)
    ecc[:,2] = ebb[-1,2]

    ucc[:,0] = np.zeros(n2, dtype=dtype)
    ucc[:,1] = np.zeros(n2, dtype=dtype)
    ucc[:,2] = np.ones(n2, dtype=dtype)

    ccc[0].view("|S2")[0:8] = ["T7","","","" ,"","","",""]
    ccc[n2//2].view("|S2")[0:8] = ["","","","" ,"","","",""]
    ccc[n2//2+2].view("|S2")[0:8] = ["","","","" ,"","","",""]

    lcc[:-1] = ecc[1:]
    lcc[-1] = [0,0,1]

    fcc = Flight(ecc,lcc,ucc,ccc) 

    ################################################## 

    f = Flight.combine( [fzz, faa, fbb, fcc] )
    elu = f.elu
    np.save("/tmp/flightpath.npy", elu ) 
    print(elu[:,3,:4].copy().view("|S2"))

    if plt:  
        f.quiver_plot(ax, sc=sc)
        fig.show()
    pass


