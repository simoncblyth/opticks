#!/usr/bin/env python
"""


"""
import argparse, logging, os, json
log = logging.getLogger(__name__)
import numpy as np
np.set_printoptions(linewidth=200, suppress=True, precision=3)
from opticks.ana.sample import sample_trig, sample_normals, sample_reject, sample_linear

def vnorm(v):
    norm = np.sqrt((v*v).sum(axis=1))
    norm3 = np.repeat(norm, 3).reshape(-1,3)
    v /=  norm3
    return v


class InputPhotons(object):
    DEFAULT_BASE = os.path.expanduser("~/.opticks/InputPhotons")
    DTYPE = np.float64 if os.environ.get("DTYPE","np.float32") == "np.float64" else np.float32

    X = np.array( [1., 0., 0.], dtype=DTYPE ) 
    Y = np.array( [0., 1., 0.], dtype=DTYPE ) 
    Z = np.array( [0., 0., 1.], dtype=DTYPE ) 

    POSITION = [0.,0.,0.]
    TIME = 0.1
    WEIGHT = 1.
    WAVELENGTH  = 440. 

    @classmethod
    def BasePath(cls, name=None):
        if name is None:
            name = os.environ.get("OPTICKS_INPUT_PHOTON", "RandomSpherical100_f4.npy")
        pass
        return os.path.join(cls.DEFAULT_BASE, name)

    @classmethod
    def Path(cls, name, ext=".npy"):
        prec = None
        if cls.DTYPE == np.float32: prec = "_f4" 
        if cls.DTYPE == np.float64: prec = "_f8" 
        return os.path.join(cls.DEFAULT_BASE, "%s%s%s" % (name, prec, ext))


    @classmethod
    def CubeCorners(cls):
        """
        :return dir: (8,3) array of normalized direction vectors  

        000  0   (-1,-1,-1)/sqrt(3)
        001  1
        010  2 
        011  3
        100  4
        101  5
        110  6 
        111  7   (+1,+1,+1)/sqrt(3)
        """
        v = np.zeros((8, 3), dtype=cls.DTYPE)
        for i in range(8): v[i] = list(map(float,[ bool(i & 1), bool(i & 2), bool(i & 4)]))
        v = 2.*v - 1.
        return vnorm(v)

    @classmethod
    def GenerateCubeCorners(cls):
        direction = cls.CubeCorners()
        polarization = vnorm(np.cross(direction, cls.Y))
    
        p = np.zeros( (8, 4, 4), dtype=cls.DTYPE )
        n = len(p)
        p[:,0,:3] = cls.POSITION + direction  # offset start position by direction vector for easy identification purposes
        p[:,0, 3] = cls.TIME*(1. + np.arange(n))  
        p[:,1,:3] = direction 
        p[:,1, 3] = cls.WEIGHT
        p[:,2,:3] = polarization
        p[:,2, 3] = cls.WAVELENGTH  
        return p 

    @classmethod
    def OutwardsCubeCorners(cls):
        direction = cls.CubeCorners()
        polarization = vnorm(np.cross(direction, cls.Y))
    
        p = np.zeros( (8, 4, 4), dtype=cls.DTYPE )
        n = len(p)
        p[:,0,:3] = cls.POSITION + direction  # offset start position by direction vector for easy identification purposes
        p[:,0, 3] = cls.TIME*(1. + np.arange(n))  
        p[:,1,:3] = direction 
        p[:,1, 3] = cls.WEIGHT
        p[:,2,:3] = polarization
        p[:,2, 3] = cls.WAVELENGTH  
        return p 


    @classmethod
    def InwardsCubeCorners(cls, radius):
        """
        :param radius: of start position
        :return p: (8,4,4) array of photons
        """
        log.info(" radius %s " % radius )
        direction = cls.CubeCorners()
        polarization = vnorm(np.cross(-direction, cls.Y))
    
        p = np.zeros( (8, 4, 4), dtype=cls.DTYPE )
        n = len(p)
        p[:,0,:3] = radius*direction 
        p[:,0, 3] = cls.TIME*(1. + np.arange(n))  
        p[:,1,:3] = -direction 
        p[:,1, 3] = cls.WEIGHT
        p[:,2,:3] = polarization
        p[:,2, 3] = cls.WAVELENGTH  
        return p 

    @classmethod
    def Axes(cls):
        """

               Z   -X
               |  .
               | . 
               |. 
       -Y......O------ Y  1
              /.
             / .
            /  .
           X   -Z
          0
        """
        v = np.zeros((6, 3), dtype=cls.DTYPE)
        v[0] = [1,0,0]
        v[1] = [0,1,0]
        v[2] = [0,0,1]
        v[3] = [-1,0,0]
        v[4] = [0,-1,0]
        v[5] = [0,0,-1]
        return v


    @classmethod
    def GenerateAxes(cls):
        direction = cls.Axes()
        polarization = np.zeros((6, 3), dtype=cls.DTYPE)
        polarization[:-1] = direction[1:]
        polarization[-1] = direction[0]
    
        p = np.zeros( (6, 4, 4), dtype=cls.DTYPE )
        n = len(p)
        p[:,0,:3] = cls.POSITION 
        p[:,0, 3] = cls.TIME*(1. + np.arange(n))  
        p[:,1,:3] = direction 
        p[:,1, 3] = cls.WEIGHT
        p[:,2,:3] = polarization
        p[:,2, 3] = cls.WAVELENGTH  
        return p 


    @classmethod
    def Parallelize1D(cls, p, r, offset=True):
        """
        :param p: photons array of shape (num, 4, 4)
        :param r: repetition number
        :return pp:  photons array of shape (r*num, 4, 4)

        See parallel_input_photons.py for tests/plotting
        """
        if r == 0:
            return p 
        pass
        o = len(p)          # original number of photons  
        pp = np.repeat(p, r, axis=0).reshape(-1,r,4,4)  # shape (8,10,4,4)

        if offset:
            for i in range(o):
                dir = p[i,1,:3]
                pol = p[i,2,:3]   # original polarization, a transverse offset direction vector
                oth = np.cross(pol, dir)
                for j in range(r):
                    jj = j - r//2
                    pp[i,j,0,:3] = jj*oth 
                pass
            pass
        pass
        return pp.reshape(-1,4,4) 


    @classmethod
    def Parallelize2D(cls, p, rr, offset=True):
        """
        :param p: original photons, shaped (o,4,4)
        :param [rj,rk]: 2d repeat dimension list 
        :return pp: shaped (o*rj*rk,4,4)  

        See parallel_input_photons.py for tests/plotting
        """
        if len(rr) != 2:
            return p 
        pass
        rj, rk = rr[0],rr[1]
        o = len(p)          # original number of photons  
        pp = np.repeat(p, rj*rk, axis=0).reshape(-1,rj,rk,4,4) 

        if offset:
            for i in range(o):
                dir = p[i,1,:3]
                pol = p[i,2,:3]           # original polarization, a transverse offset direction vector
                oth = np.cross(pol, dir)  # other transverse direction perpendicular to pol 
                for j in range(rj):
                    jj = j - rj//2
                    for k in range(rk):
                        kk = k - rk//2
                        pp[i,j,k,0,:3] = jj*oth + kk*pol
                    pass
                pass
            pass
        pass
        return pp.reshape(-1,4,4) 

    @classmethod
    def GenerateXZ(cls, n, mom):
        """
  
               +-----------------------------------+
               |                                   |
               |                                   |
               |                                   |
               |                                   |
               |                                   |
               |                                   |
               |                                   |
               |                                   |
               |                                   |
               |                                   |
               |                                   |         Z
               |          ^ ^ ^ ^ ^ ^ ^            |         |  Y
               |          | | | | | | |            |         | /
               |          . . . . . . .            |         |/
               +-----------------------------------+         +---> X
             -100        -49    0     49          100               


        """
        pos = np.zeros((n,3), dtype=cls.DTYPE )
        pos[:,0] = sample_linear(n, -49., 49. )
        pos[:,1] = 0.
        pos[:,2] = -99. 

        p = np.zeros( (n, 4, 4), dtype=cls.DTYPE )
        p[:,0,:3] = pos
        p[:,0, 3] = cls.TIME
        p[:,1,:3] = mom           # mom : Up:self.Z or Down:-self.Z  
        p[:,1, 3] = cls.WEIGHT 
        p[:,2,:3] = cls.Y         # pol
        p[:,2, 3] = cls.WAVELENGTH  
        return p  

    @classmethod
    def GenerateRandomSpherical(cls, n):
        """
        :param n: number of photons to generate

        spherical distribs not carefully checked  

        The start position is offset by the direction vector for easy identification purposes
        so that means the rays will start on a virtual unit sphere and travel radially 
        outwards from there.

        """
        spherical = sample_trig(n).T
        assert spherical.shape == (n,3)

        direction = spherical
        polarization = vnorm(np.cross(direction,cls.Y))

        p = np.zeros( (n, 4, 4), dtype=cls.DTYPE )
        p[:,0,:3] = cls.POSITION + direction 
        p[:,0, 3] = cls.TIME*(1. + np.arange(n))  
        p[:,1,:3] = direction 
        p[:,1, 3] = cls.WEIGHT 
        p[:,2,:3] = polarization
        p[:,2, 3] = cls.WAVELENGTH  
        return p 

    @classmethod
    def GenerateRandomDisc(cls, n):
        spherical = sample_trig(n).T
        disc_offset = spherical.copy() 
        disc_offset[:,0] *= 100. 
        disc_offset[:,1] *= 100. 
        disc_offset[:,2] = 0. 

        p = np.zeros( (n, 4, 4), dtype=cls.DTYPE )
        p[:,0,:3] = cls.POSITION + disc_offset
        p[:,0, 3] = cls.TIME*(1. + np.arange(n))  
        p[:,1,:3] = cls.Z
        p[:,1, 3] = cls.WEIGHT 
        p[:,2,:3] = cls.X
        p[:,2, 3] = cls.WAVELENGTH  
        return p 

    @classmethod
    def CheckTransverse(cls, direction, polarization, epsilon):
        # check elements should all be very close to zero
        check1 = np.einsum('ij,ij->i',direction,polarization)  
        check2 = (direction*polarization).sum(axis=1)
        assert np.abs(check1).min() < epsilon
        assert np.abs(check2).min() < epsilon

    @classmethod
    def Check(cls, p):
        direction = p[:,1,:3]
        polarization = p[:,2,:3]
        cls.CheckTransverse( direction, polarization, 1e-6 )

    CC = "CubeCorners" 
    ICC = "InwardsCubeCorners" 
    RS = "RandomSpherical" 
    RD = "RandomDisc" 
    UXZ = "UpXZ"
    DXZ = "DownXZ"

    NAMES = [CC, CC+"10x10", CC+"100", CC+"100x100", RS+"10", RS+"100", ICC+"17699", ICC+"1", RD+"10", RD+"100", UXZ+"1000", DXZ+"1000" ]

    def generate(self, name, args):
        if args.seed > -1:
            log.info("seeding with %d " % args.seed)
            np.random.seed(args.seed)
        pass
        meta = dict(seed=args.seed, name=name, creator="input_photons.py")
        log.info("generate %s " % name)
        if name.startswith(self.RS):  
            num = int(name[len(self.RS):])
            p = self.GenerateRandomSpherical(num)    
        elif name.startswith(self.UXZ) or name.startswith(self.DXZ):  
            num = None
            mom = None
            if name.startswith(self.UXZ): 
                num = int(name[len(self.UXZ):])
                mom = self.Z 
            elif name.startswith(self.DXZ): 
                num = int(name[len(self.DXZ):])
                mom = -self.Z 
            else:
                pass
            pass
            p = self.GenerateXZ(num, mom)    
        elif name.startswith(self.RD):  
            num = int(name[len(self.RD):])
            p = self.GenerateRandomDisc(num)    
        elif name == self.CC:
            p = self.GenerateCubeCorners()    
        elif name.startswith(self.CC):
            o = self.OutwardsCubeCorners()    
            sdim = name[len(self.CC):]
            if sdim.find("x") > -1:
                rr = list(map(int, sdim.split("x")))
                p = self.Parallelize2D(o, rr)
                meta["Parallelize2D_rr"] = rr 
            else:
                r = int(sdim)
                p = self.Parallelize1D(o, r)
                meta["Parallelize1D_r"] = r 
            pass 
        elif name.startswith(self.ICC):
            sradius = name[len(self.ICC):]
            radius = float(sradius)
            p = self.InwardsCubeCorners(radius)             
        else:
            log.fatal("no generate method for name %s " %  name)
            assert 0
        pass     
        self.Check(p)
        meta.update(num=len(p))
        return p, meta 


    def __init__(self, name, args=None):
        if args is None:
            args = InputPhotonDefaults()
        pass
        npy_path = self.Path(name, ext=".npy")
        json_path = self.Path(name, ext=".json")
        generated = False
        if os.path.exists(npy_path) and os.path.exists(json_path):
            log.info("load %s from %s %s " % (name, npy_path, json_path))
            p = np.load(npy_path)
            meta = json.load(open(json_path,"r"))
        else: 
            p, meta = self.generate(name, args)
            generated = True
        pass
        self.p = p 
        self.meta = meta
        if generated:
            self.save()
        pass

    name = property(lambda self:self.meta.get("name", "no-name"))

    def save(self):
        npy_path = self.Path(self.name, ext=".npy")
        json_path = self.Path(self.name, ext=".json")
        fold = os.path.dirname(npy_path)
        if not os.path.isdir(fold):
            log.info("creating folder %s " % fold) 
            os.makedirs(fold)
        pass
        log.info("save %s to %s and %s " % (self.name, npy_path, json_path))
        np.save(npy_path, self.p)
        json.dump(self.meta, open(json_path,"w"))

    def __repr__(self):
        return "\n".join([str(self.meta),str(self.p.reshape(-1,16))])

    @classmethod
    def parse_args(cls, doc, names):
        defaults = InputPhotonDefaults
        parser = argparse.ArgumentParser(doc)
        parser.add_argument( "names", nargs="*", default=names, help="Name stem of InputPhotons array, default %(default)s" )
        parser.add_argument( "--level", default=InputPhotonDefaults.level, help="logging level, default %(default)s" ) 
        parser.add_argument( "--seed", type=int, default=InputPhotonDefaults.seed, help="seed for np.random.seed() or -1 for non-reproducible generation, default %(default)s" ) 
        args = parser.parse_args()
        return args 

class InputPhotonDefaults(object):
    seed = 0 
    level = "info"

if __name__ == '__main__':
    args = InputPhotons.parse_args(__doc__, InputPhotons.NAMES)

    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)

    ip = {}
    for name in args.names:
        ip[name] = InputPhotons(name, args)
        print(ip[name])
    pass

    sel = "InwardsCubeCorners17699"
    ip0 = ip[sel] 
    p = ip0.p 
    m = ip0.meta
    r = np.sqrt(np.sum(p[:,0,:3]*p[:,0,:3], axis=1 ))  # radii of start positions





