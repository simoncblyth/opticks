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
makeflight.py : creates eye-look-up input NumPy arrays used by okc/FlightPath.cc  
===================================================================================

See docs/misc/making_flightpath_raytrace_movies.rst 

"""
import os, inspect, logging, argparse, numpy as np
log = logging.getLogger(__name__)

dtype = np.float32

class Flight(object):
    DEFAULT_BASE = os.path.expanduser("~/.opticks/flight")

    @classmethod
    def Roundabout(cls, plane='XY', scale=1, steps=32):
        """

        Move eye in circle in XY/ZX/YZ plane whilst looking towards the center, up +Z/+Y/+X

        **plane XY, +Z up**                          **plane ZX, +Y up**                  
                                                                          
                                                      X
                 Y                                   3|
                3|                                    |   2  
                 |   2                                | 
                 |                                    |      1
                 |      1                             |
                 |                                    O--------0----Z
                 O--------0----X                     /
                /                                   /
               /                                   /
              /                                   Y                      
             Z                       

         **plane YZ, +X up**

                 Z
                3|
                 |   2  
                 | 
                 |      1
                 |
                 O--------0----Y
                /
               /
              /
             X                      

        Note that the flightpath can be scaled after loading 
        into Opticks executables using the --flightpathscale option.

        Its generally more flexible to change scale that way, avoiding 
        the need to recreate flightpath.npy files for simple scale changes.
        """

        method = inspect.currentframe().f_code.co_name
        name = "{method}{plane}".format(**locals())

        log.info("plane %s scale %s steps %s name %s " % (plane, scale, steps, name))

        f = cls.Make( name, steps) 
        ta = np.linspace( 0, 2*np.pi, steps )
        st = np.sin(ta)
        ct = np.cos(ta)
        n = len(ta)

        if plane == 'XY':  
            f.e[:,0] = ct*scale  # X         
            f.e[:,1] = st*scale  # Y
            f.e[:,2] = 0
            f.e[:,3] = 1

            f.l[:] = [0,0,0,1]   # always looking at center 
            f.u[:] = [0,0,1,0]   # always up +Z

        elif plane == 'ZX':
           
            f.e[:,0] = st*scale  # X         
            f.e[:,1] = 0
            f.e[:,2] = ct*scale  # Z
            f.e[:,3] = 1

            f.l[:] = [0,0,0,1]   # always looking at center 
            f.u[:] = [0,1,0,0]   # always up +Y

        elif plane == 'YZ':

            f.e[:,0] = 0         
            f.e[:,1] = ct*scale  # Y
            f.e[:,2] = st*scale  # Z 
            f.e[:,3] = 1

            f.l[:] = [0,0,0,1]   # always looking at center 
            f.u[:] = [1,0,0,0]   # always up +X
        else:
            pass
        pass
        return f

    @classmethod
    def Path(cls, name):
        return os.path.join(cls.DEFAULT_BASE, "%s.npy" % name)

    @classmethod
    def Make(cls, name, n):
        eluc = np.zeros( (n,4,4), dtype=np.float32)
        return cls(name, eluc)

    @classmethod
    def Load(cls, name):
        path = cls.Path(name)
        eluc = np.load(path)
        return cls(name, eluc)

    @classmethod
    def Combine(cls, names, combined_name):
        arrs = []
        for name in names:
            path = cls.Path(name)
            arr = np.load(path)
            print( " %15s : %s " % (str(arr.shape), path ))
            arrs.append(arr)  
        pass
        return cls.CombineArrays(arrs, combined_name)

    @classmethod
    def CombineArrays(cls, arrs, combined_name):
        eluc = np.concatenate(tuple(arrs))  
        return cls(combined_name, eluc)

    def __init__(self, name, eluc ):
        self.name = name
        self.eluc = eluc

    e = property(lambda self:self.eluc[:,0,:4] ) 
    l = property(lambda self:self.eluc[:,1,:4] ) 
    u = property(lambda self:self.eluc[:,2,:4] ) 
    c = property(lambda self:self.eluc[:,3,:4] ) 

    def save(self):
        path = self.Path(self.name)
        fold = os.path.dirname(path)
        if not os.path.isdir(fold):
            log.info("creating directory %s " % fold)
            os.makedirs(fold)
        pass 
        log.info("saving to %s " % path )
        np.save(path, self.eluc )

    def print_cmds(self):
        print(self.c.copy().view("|S2"))

    def __len__(self):
        return len(self.eluc) 

    def __repr__(self):
        return "Flight %s eluc.shape %s " % (self.name, str(self.eluc.shape))


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


    f = {}
    for p in ['XY', 'ZX', 'YZ' ]:
        f[p] = Flight.Roundabout(plane=p, scale=args.scale, steps=args.steps)
    pass

    p = 'XY_ZX_YZ'
    f[p] = Flight.CombineArrays([f['XY'].eluc, f['ZX'].eluc,f['YZ'].eluc], 'Roundabout_%s' % p )

    for p in f.keys():
        print(f[p])
        f[p].save()
    pass

