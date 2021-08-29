#/usr/bin/env python

import os, numpy as np, logging
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt 
from opticks.ana.main import opticks_main 
from opticks.ana.key import keydir

def test_digitize():
    edom = np.linspace(-1., 12., 131 )
    ebin = np.linspace( 1., 10., 10 )
    idom = np.digitize(edom, ebin, right=False)  # right==False (the default) indicates the interval does not include the right edge
    # below and above yields 0 and len(ebin) = ebin.shape[0]
    #print(np.c_[edom,idom])    
    for e, i in zip(edom,idom):
        mkr = "---------------------------" if e in ebin else ""
        print(" %7.4f : %d : %s " % (e, i, mkr))
    pass



class S2(object):
    def __init__(self):
        kd = keydir(os.environ["OPTICKS_KEY"])
        ri = np.load(os.path.join(kd, "GScintillatorLib/LS_ori/RINDEX.npy"))
        ri[:,0] *= 1e6  
        ri_ = lambda e:np.interp(e, ri[:,0], ri[:,1] )

        nmax = ri[:,1].max() 
        nmin = ri[:,1].min() 

        ebin = ri[:,0] 
        efin = np.linspace( ebin[0], ebin[-1], 280 )

        self.ri = ri 
        self.ri_ = ri_

        self.ebin = ebin 
        self.efin = efin 


    def bin(self, idx, lhs=1., rhs=1.):
        """
        Example, bins array of shape (4,) with 3 bins::

         |       +-------------+--------------+-------------+         |
              0        1             2               3            4 


        Hmm under/over bins are dangerous for "inflation"
        """
        dom = self.ri[:,0]
        val = self.ri[:,1]

        d = None
        v = None
        if idx == 0:
            d = [dom[0] - lhs, dom[0] ]
            v = [val[0], val[0]]
        elif idx > 0 and idx < len(dom):
            d = [dom[idx-1], dom[idx] ]
            v = [val[idx-1], val[idx]]
        elif idx == len(dom):
            d = [dom[idx-1], dom[idx-1] + rhs]
            v = [val[idx-1], val[idx-1]]
        pass
        return np.array(d), np.array(v)

        
    def bin_dump(self, BetaInverse=1.5):
        """
        """
        for idx in range(len(self.ri)+1):
            en, ri = self.bin(idx) 
            s2i = self.bin_integral(idx, BetaInverse)
            print( " idx %3d  en %20s   ri %20s  s2i %10.4f  " % (idx, str(en), str(ri), s2i ))
        pass
 
    def bin_integral(self, idx, BetaInverse): 
        """
        :param idx: 0-based bin index  
        :param BetaInverse: scalar
        :return s2integral: scalar sin^2 ck_angle integral in energy range  
        """
        en, ri = self.bin(idx) 
        ct = BetaInverse/ri
        s2 = (1.-ct)*(1.+ct) 

        en_0,en_1 = en
        s2_0,s2_1 = s2

        ret = 0.
        if s2_0 <= 0. and s2_1 <= 0.: 
            ret = 0.
        elif s2_0 < 0. and s2_1 > 0.:    # s2 goes +ve 
            en_cross = (s2_1*en_0 - s2_0*en_1)/(s2_1 - s2_0)
            ret = (en_1 - en_cross)*s2_1*0.5
        elif s2_0 >= 0. and s2_1 >= 0.:   # s2 +ve or zero across full range 
            ret = (en_1 - en_0)*(s2_0 + s2_1)*0.5
        elif s2_0 > 0. and s2_1 <= 0.:     # s2 goes -ve or to zero 
            en_cross = (s2_1*en_0 - s2_0*en_1)/(s2_1 - s2_0) 
            ret = (en_cross - en_0)*s2_0*0.5
        else:
            assert 0 
        pass
        assert ret >= 0.  
        return ret 

    def bin_integral_check(self):
        bis = np.linspace(1, 2, 11)
        for BetaInverse in bis:
            en = (self.ebin[0], self.ebin[-1])
            s2i = self.bin__integral(en, BetaInverse)
            print(" bi %7.4f s2i  %7.4f " % (BetaInverse, s2i ))
        pass

        


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    s2 = S2()
    s2.bin_dump() 
 
