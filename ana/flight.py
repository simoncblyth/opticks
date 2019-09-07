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
import os, logging, numpy as np
log = logging.getLogger(__name__)

dtype = np.float32

class Flight(object):
    @classmethod
    def combine(cls, ff):
        na = sum(map(len,ff))
        e = np.zeros( (na, 3), dtype=dtype )
        l = np.zeros( (na, 3), dtype=dtype )
        u = np.zeros( (na, 3), dtype=dtype )
        c = np.zeros( (na, 4), dtype=dtype )

        n0 = 0 
        for f in ff: 
            nf = len(f)
            e[n0:n0+nf] = f.e
            l[n0:n0+nf] = f.l
            u[n0:n0+nf] = f.u
            c[n0:n0+nf] = f.c
            n0 += nf
        pass  
        return cls(e,l,u,c)

    def __init__(self, e, l, u, c):
        assert len(e) == len(l) == len(u) == len(c) 
        self.e = e
        self.l = l
        self.u = u
        self.c = c

    def _get_elu(self):
        elu = np.zeros( (len(self),4,4), dtype=np.float32)
        elu[:,0,:3] = self.e 
        elu[:,1,:3] = self.l
        elu[:,2,:3] = self.u
        elu[:,3,:4] = self.c
        return elu
    elu = property(_get_elu)

    def __len__(self):
        return len(self.e) 

    def quiver_plot(self, ax, sc):
        e = self.e 
        l = self.l
        u = self.u 
        g = l - e

        x = sc*e[:,0] 
        y = sc*e[:,1] 
        z = sc*e[:,2]

        u0 = g[:, 0] 
        v0 = g[:, 1] 
        w0 = g[:, 2] 

        u1 = u[:, 0] 
        v1 = u[:, 1] 
        w1 = u[:, 2] 
  
        #ax.plot( x,z )
        ax.quiver( x, y, z, u0, v0, w0  ) 
        ax.quiver( x, y, z, u1, v1, w1  ) 

        labels = False
        if labels:
            for i in range(len(e)):
                ax.text( x[i], y[i], z[i], i , "z" )
            pass  
        pass


if __name__ == '__main__':
    pass
    np.set_printoptions(suppress=True)





