#!/usr/bin/env python
"""


"""
import argparse, logging 
log = logging.getLogger(__name__)
import numpy as np
from opticks.ana.sample import sample_trig, sample_normals, sample_reject


class Generator(object):
    def __init__(self, n=5 ):

        # random spherical distribs : distrib not carefully checked  
        a = sample_trig(n)
        b = sample_normals(n)
        c = sample_reject(n)

        position_time = [0,0,0,0.1] 
        direction = a.T

        x = np.array( [1., 0., 0.], dtype=np.float32 ) 
        y = np.array( [0., 1., 0.], dtype=np.float32 ) 
        z = np.array( [0., 0., 1.], dtype=np.float32 ) 

        pol = np.cross(direction, y) 
        mpol = np.sqrt(np.sum(pol*pol, axis=1 ))
        mpol3 = np.repeat(mpol, 3).reshape(-1,3)

        polarization = pol/mpol3

        # check elements should all be very close to zero
        check1 = np.einsum('ij,ij->i',direction,polarization)  
        check2 = (direction*polarization).sum(axis=1)

        weight = 1.
        wavelength = 440. 

        ph = np.zeros( (n, 4, 4), dtype=np.float32 )
        ph[:,0] = position_time
        ph[:,1,:3] = direction 
        ph[:,1, 3] = weight 
        ph[:,2,:3] = polarization
        ph[:,2, 3] = wavelength  

        self.ph = ph

    def save(self, path):
        log.info("saving to %s shape %s " % (path, str(self.ph.shape)))
        print(self.ph)
        np.save(path, self.ph)



def parse_args(doc):
    parser = argparse.ArgumentParser(doc)
    parser.add_argument( "--path", default="/tmp/input_photons.npy", help="Path where input_photons array will be written, default %(default)s" )
    parser.add_argument( "--num", type=int, default=10, help="number of photons to generate, default %(default)s" ) 
    parser.add_argument( "--level", default="info", help="logging level, default %(default)s" ) 
    args = parser.parse_args()
    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
    return args 


if __name__ == '__main__':
    args = parse_args(__doc__)
    gen = Generator(args.num)
    gen.save(args.path)



