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
flight.py
===========

See docs/misc/making_flightpath_raytrace_movies.rst 


"""
import os, logging, argparse, numpy as np
log = logging.getLogger(__name__)

dtype = np.float32

class Flight(object):

    DEFAULT_PATH = "/tmp/flightpath.npy"


    @classmethod
    def RoundaboutXY(cls, scale=1, steps=32):
        """
        Move eye in circle in XY plane whilst looking towards the center, up +Z

        
                 Y
                3|
                 |   2  
                 | 
                 |      1
                 |
                 +--------0----X 


        Note that the flightpath can be scaled after loading 
        into Opticks executables using the --flightpathscale option.

        Its generally more flexible to change scale that way, avoiding 
        the need to recreate flightpath.npy files for simple scale changes.
        """
        log.info("scale %s steps %s " % (scale, steps))

        ta = np.linspace( 0, 2*np.pi, steps )
        st = np.sin(ta)
        ct = np.cos(ta)
        n = len(ta)

        e = np.zeros( (n,4) , dtype=dtype )  # eye position 
        l = np.zeros( (n,4) , dtype=dtype )  # look position
        u = np.zeros( (n,4) , dtype=dtype )  # up direction
        c = np.zeros( (n,4),  dtype=dtype )  # ctrl  

        e[:,0] = ct*scale           
        e[:,1] = st*scale
        e[:,2] = 0
        e[:,3] = 1

        l[:] = [0,0,0,1]   # always looking at center 
        u[:] = [0,0,1,0]   # always up +Z

        return cls(e,l,u,c)

    @classmethod
    def combine(cls, ff):
        na = sum(map(len,ff))
        e = np.zeros( (na, 4), dtype=dtype )
        l = np.zeros( (na, 4), dtype=dtype )
        u = np.zeros( (na, 4), dtype=dtype )
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
        elu[:,0,:4] = self.e 
        elu[:,1,:4] = self.l
        elu[:,2,:4] = self.u
        elu[:,3,:4] = self.c
        return elu
    elu = property(_get_elu)

    def save(self, path=DEFAULT_PATH):
        np.save(path, self.elu )

    @classmethod
    def Load(cls, path=DEFAULT_PATH):
        a = np.load(path)
        e = a[:,0,:4]
        l = a[:,1,:4]
        u = a[:,2,:4]
        c = a[:,3,:4]
        return cls(e, l, u, c )

    def print_cmds(self):
        print(self.elu[:,3,:4].copy().view("|S2"))

    def __len__(self):
        return len(self.e) 

    def __repr__(self):
        return "Flight elu.shape %s " % str(self.elu.shape)

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




def parse_args(doc, **kwa):
    np.set_printoptions(suppress=True, precision=3, linewidth=200)
    parser = argparse.ArgumentParser(doc)
    parser.add_argument( "--level", default="info", help="logging level" ) 
    parser.add_argument( "--roundaboutxy", action="store_true", help="" ) 
    parser.add_argument( "--steps", default=32, type=int, help="Number of steps in flightpath that are interpolated between in InterpolatedView " ) 
    parser.add_argument( "--scale", default=1, type=float, help="scale of the flightpath, for example the radius for circles" ) 
    args = parser.parse_args()
    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
    return args  


if __name__ == '__main__':
    pass
    np.set_printoptions(suppress=True)
    args = parse_args(__doc__)
    if args.roundaboutxy:
        f = Flight.RoundaboutXY(scale=args.scale, steps=args.steps)
        print(f)
        f.save()
    pass



