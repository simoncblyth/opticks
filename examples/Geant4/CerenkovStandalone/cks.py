#!/usr/bin/env python
"""

::

    ipython -i cks.py 

"""
import os, logging, numpy as np
from opticks.ana.key import keydir
log = logging.getLogger(__name__)

class CKS(object):
    PATH="/tmp/cks/cks.npy" 

    kd = keydir()
    rindex_path = os.path.join(kd, "GScintillatorLib/LS_ori/RINDEX.npy")
    random_path = os.path.expandvars("/tmp/$USER/opticks/TRngBufTest_0.npy")

    def __init__(self):
        rnd = np.load(self.random_path) 
        num = len(rnd)
        cursors = np.zeros( num, dtype=np.int32 ) 

        self.rnd = rnd 
        self.num = num 


        rindex = np.load(self.rindex_path)
        rindex[:,0] *= 1e6    # into eV 
        rindex_ = lambda ev:np.interp( ev, rindex[:,0], rindex[:,1] ) 

        Pmin = rindex[0,0]  
        Pmax = rindex[-1,0]  
        nMax = rindex[:,1].max() 

        self.rindex = rindex
        self.Pmin = Pmin
        self.Pmax = Pmax
        self.nMax = nMax
        
        self.rindex_ = rindex_
        self.cursors = cursors
        self.p = np.zeros( (num,4,4), dtype=np.float64 )

    def energy_sample_all(self, BetaInverse=1.5):
        for idx in range(self.num):
            self.energy_sample(idx, BetaInverse=BetaInverse) 
        pass

    def energy_sample(self, idx, BetaInverse=1.5):
        rnd = self.rnd
        rindex = self.rindex
        rindex_ = self.rindex_
        cursors = self.cursors  
        num = self.num
        Pmin = self.Pmin
        Pmax = self.Pmax  
        nMax = self.nMax

        uu = rnd[idx].ravel()
        maxCos = BetaInverse / nMax
        maxSin2 = (1.0 - maxCos) * (1.0 + maxCos)

        dump = idx < 10 or idx > num - 10  
        loop = 0 
 
        while True:
            u0 = uu[cursors[idx]]
            cursors[idx] += 1 

            u1 = uu[cursors[idx]]
            cursors[idx] += 1 

            sampledEnergy = Pmin + u0*(Pmax-Pmin)
            sampledRI = rindex_(sampledEnergy)
            cosTheta = BetaInverse/sampledRI
            sin2Theta = (1.-cosTheta)*(1.+cosTheta)

            u1_maxSin2 = u1*maxSin2
            keep_sampling = u1_maxSin2 > sin2Theta

            loop += 1  

            if dump:
                fmt = "idx %5d u0 %10.5f sampledEnergy %10.5f sampledRI %10.5f cosTheta %10.5f sin2Theta %10.5f u1 %10.5f"
                vals = (idx, u0, sampledEnergy, sampledRI, cosTheta, sin2Theta, u1 )
                print(fmt % vals) 
            pass

            if not keep_sampling:
                break
            pass
        pass

        hc_eVnm = 1239.8418754200 # G4: h_Planck*c_light/(eV*nm)   

        sampledWavelength = hc_eVnm/sampledEnergy 

        p = self.p[idx]  
        i = self.p[idx].view(np.uint64) 

        p[0,0] = sampledEnergy
        p[0,1] = sampledWavelength
        p[0,2] = sampledRI
        p[0,3] = cosTheta

        p[1,0] = sin2Theta 
        i[3,1] = loop

    def save(self):
        fold = os.path.dirname(self.PATH)
        if not os.path.exists(fold):
            os.makedirs(fold)
        pass
        log.info("save to %s " % self.PATH)
        np.save(self.PATH, self.p)

    @classmethod
    def Load(cls):
        return np.load(cls.PATH)

 

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)


    cks = CKS()
    cks.energy_sample_all(BetaInverse=1.5)
    cks.save()

    p = cks.p



