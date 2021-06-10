#!/usr/bin/env python
"""


"""
import argparse, logging, os, json
log = logging.getLogger(__name__)
import numpy as np
np.set_printoptions(linewidth=200, suppress=True, precision=3)
from opticks.ana.sample import sample_trig, sample_normals, sample_reject

def vnorm(v):
    norm = np.sqrt((v*v).sum(axis=1))
    norm3 = np.repeat(norm, 3).reshape(-1,3)
    v /=  norm3
    return v


class InputPhotons(object):
    DEFAULT_BASE = os.path.expanduser("~/.opticks/InputPhotons")

    DTYPE = np.float32

    X = np.array( [1., 0., 0.], dtype=DTYPE ) 
    Y = np.array( [0., 1., 0.], dtype=DTYPE ) 
    Z = np.array( [0., 0., 1.], dtype=DTYPE ) 

    POSITION = [0.,0.,0.]
    TIME = 0.1
    WEIGHT = 1.
    WAVELENGTH  = 440. 

    @classmethod
    def Path(cls, name, ext=".npy"):
        return os.path.join(cls.DEFAULT_BASE, "%s%s" % (name, ext))

    @classmethod
    def CubeCorners(cls):
        """
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
    def GenerateRandomSpherical(cls, n):
        """
        spherical distribs not carefully checked  
        """
        spherical = sample_trig(n).T
        direction = spherical
        polarization = vnorm(np.cross(direction,cls.Y))

        p = np.zeros( (n, 4, 4), dtype=cls.DTYPE )
        p[:,0,:3] = cls.POSITION + direction  # offset start position by direction vector for easy identification purposes
        p[:,0, 3] = cls.TIME*(1. + np.arange(n))  
        p[:,1,:3] = direction 
        p[:,1, 3] = cls.WEIGHT 
        p[:,2,:3] = polarization
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
    RS = "RandomSpherical" 
    NAMES = [CC, CC+"10x10", CC+"100", CC+"100x100", RS+"10"]

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
        elif name == self.CC:
            p = self.GenerateCubeCorners()    
        elif name.startswith(self.CC):
            o = self.GenerateCubeCorners()    
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


