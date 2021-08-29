#/usr/bin/env python
"""


       [ 8.857,  9.3  ,  0.188,  0.113, -1.   ,  0.067,  0.   ,  0.676,  0.743, 15.   ,  9.3  , 55.   ,  0.743],
       [ 8.857,  9.441,  0.188,  0.087, -1.   ,  0.08 ,  0.   ,  0.676,  0.756, 15.   ,  9.441, 56.   ,  0.756],

       [ 9.538,  9.582,  0.069,  0.062, -1.   ,  0.003,  0.   ,  0.763,  0.766, 16.   ,  9.582, 57.   ,  0.766],
       [ 9.538,  9.723,  0.069,  0.04 , -1.   ,  0.01 ,  0.   ,  0.763,  0.773, 16.   ,  9.723, 58.   ,  0.773],
       [ 9.538,  9.864,  0.069,  0.017, -1.   ,  0.014,  0.   ,  0.763,  0.777, 16.   ,  9.864, 59.   ,  0.777],
       [ 9.538, 10.005,  0.069, -0.007,  9.964,  0.015,  0.   ,  0.763,  0.778, 16.   , 10.005, 60.   ,  0.778],
       [ 9.538, 10.145,  0.069, -0.031,  9.956, *0.014*, 0.   ,  0.763,  0.778, 16.   , 10.145, 61.   ,  0.778],
       [ 9.538, 10.286,  0.069, -0.057,  9.948,  0.014,  0.   ,  0.763,  0.778, 16.   , 10.286, 62.   ,  0.778],

       [10.33 , 10.427, -0.065, -0.065, -1.   ,  0.   ,  0.   ,  0.777,  0.777, 17.   , 10.427, 63.   ,  0.777],
       [10.33 , 10.568, -0.065, -0.065, -1.   ,  0.   ,  0.   ,  0.777,  0.777, 17.   , 10.568, 64.   ,  0.777],
       [10.33 , 10.709, -0.065, -0.065, -1.   ,  0.   ,  0.   ,  0.777,  0.777, 17.   , 10.709, 65.   ,  0.777],
       [10.33 , 10.85 , -0.065, -0.065, -1.   ,  0.   ,  0.   ,  0.777,  0.777, 17.   , 10.85 , 66.   ,  0.777],
       [10.33 , 10.991, -0.065, -0.065, -1.   ,  0.   ,  0.   ,  0.777,  0.777, 17.   , 10.991, 67.   ,  0.777],
       [10.33 , 11.132, -0.065, -0.065, -1.   ,  0.   ,  0.   ,  0.777,  0.777, 17.   , 11.132, 68.   ,  0.777],
       [10.33 , 11.273, -0.065, -0.065, -1.   ,  0.   ,  0.   ,  0.777,  0.777, 17.   , 11.273, 69.   ,  0.777],


        en_0     en_1    s2_0    s2_1

In [25]: np.c_[s2.ri, np.arange(len(s2.ri))+1 ]
Out[25]: 
array([[ 1.55 ,  1.478,  1.   ],
       [ 1.795,  1.48 ,  2.   ],
       [ 2.105,  1.484,  3.   ],
       [ 2.271,  1.486,  4.   ],
       [ 2.551,  1.492,  5.   ],
       [ 2.845,  1.496,  6.   ],
       [ 3.064,  1.499,  7.   ],
       [ 4.133,  1.526,  8.   ],
       [ 6.2  ,  1.619,  9.   ],
       [ 6.526,  1.618, 10.   ],
       [ 6.889,  1.527, 11.   ],
       [ 7.294,  1.554, 12.   ],
       [ 7.75 ,  1.793, 13.   ],
       [ 8.267,  1.783, 14.   ],
       [ 8.857,  1.664, 15.   ],

       [ 9.538,  1.554, 16.   ],
       [10.33 ,  1.454, 17.   ],

       [15.5  ,  1.454, 18.   ]])



"""

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
        self.ri = ri 
        self.ri_ = ri_
        self.s2x = np.repeat( np.nan, len(self.ri) )


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
            s2i, branch, en_cross, meta = self.bin_integral(idx, BetaInverse)
            print( " idx %3d  en %20s   ri %20s  s2i %10.4f  " % (idx, str(en), str(ri), s2i ))
        pass


    def linear_integral(self, en_0, en_1, s2_0, s2_1, fix_en_cross=None ):
        """
        :param en_0:
        :param en_1:
        :param s2_0:
        :param s2_1:
        :param fix_en_cross:

        Consider integrating in a region close to 
        where s2 goes from +ve to -ve with en_0 with s2_0 > 0 
        
        As the en_1 is increased beyond the crossing where s2 is zero 
        the calculation of en_cross will vary a little 
        depending on en_1 and s2_1 despite the expectation 
        that extending the en_1 shouuld not be adding to the integral.
        
        This is problematic for forming the CDF because it may cause
        the result of the cumulative integral to become slightly non-monotonic.

        The root of the problem is repeated calculation of en_cross
        as the en_1 in increased despite there being no additional info 
        only chance to get different en_cross from linear extrapolation numerical 
        imprecision.

        For each crossing bin need to calulate the en_cross once and reuse that.
        Then can use fix_en_cross optional argument to ensure that the 
        same en_cross is used for all integrals relevant to a bin.


             .  
             |  .
             |     .
             +-------X----------
                        .
                           .

        """
        ret = 0.
        en_cross = np.nan
        branch = -1
        if s2_0 <= 0. and s2_1 <= 0.: 
            ret = 0.
            branch = 1
        elif s2_0 < 0. and s2_1 > 0.:    # s2 goes +ve 
            en_cross = (s2_1*en_0 - s2_0*en_1)/(s2_1 - s2_0) if fix_en_cross is None else fix_en_cross
            ret = (en_1 - en_cross)*s2_1*0.5
            branch = 2
        elif s2_0 >= 0. and s2_1 >= 0.:   # s2 +ve or zero across full range 
            ret = (en_1 - en_0)*(s2_0 + s2_1)*0.5
            branch = 3
        elif s2_0 > 0. and s2_1 <= 0.:     # s2 goes -ve or to zero 
            en_cross = (s2_1*en_0 - s2_0*en_1)/(s2_1 - s2_0) if fix_en_cross is None else fix_en_cross
            ret = (en_cross - en_0)*s2_0*0.5
            branch = 4
        else:
            assert 0 
        pass
        meta = np.zeros(12)

        meta[0] = en_0
        meta[1] = en_1
        meta[2] = s2_0
        meta[3] = s2_1

        meta[4] = en_cross
        meta[5] = ret 
        meta[6] = 0.
        meta[7] = 0.

        meta[8] = 0.
        meta[9] = 0.
        meta[10] = 0.
        meta[11] = 0.

        #assert ret >= 0.  
        return ret, en_cross, branch, meta 

    def s2_linear_integral(self, en_, ri_, BetaInverse, fix_en_cross=None):

        en = np.asarray(en_)
        ri = np.asarray(ri_)

        ct = BetaInverse/ri
        s2 = (1.-ct)*(1.+ct) 

        en_0, en_1 = en
        s2_0, s2_1 = s2

        return self.linear_integral( en_0, en_1, s2_0, s2_1, fix_en_cross=fix_en_cross )
 
    def bin_integral(self, idx, BetaInverse, fix_en_cross=None): 
        """
        :param idx: 0-based bin index  
        :param BetaInverse: scalar
        :return s2integral: scalar sin^2 ck_angle integral in energy range  
        """
        en, ri = self.bin(idx) 
        return self.s2_linear_integral(en, ri, BetaInverse, fix_en_cross=fix_en_cross )

    def bin_integral_check(self):
        bis = np.linspace(1, 2, 11)
        for BetaInverse in bis:
            en = (self.ebin[0], self.ebin[-1])
            s2i, en_cross, branch, meta = self.bin_integral(en, BetaInverse)
            print(" bi %7.4f s2i  %7.4f branch %d en_cross %7.4f  " % (BetaInverse, s2i, branch, en_cross ))
        pass

    def ecut_integral(self, line, ecut, BetaInverse=1.5, dump=False):
        """
        """
        ## find index of the bin containing the ecut value  
        idx = int(np.digitize( ecut, self.ri[:,0], right=False ))

        s2integral = np.zeros(len(self.ri)+1)

        ## first add up integrals from full bins to the left of idx bin 
        for i in range(1,idx):  
            s2integral[i], en_cross, branch,  _ = self.bin_integral(i, BetaInverse)
        pass

        ## find details of the idx bin 
        en, ri = self.bin(idx)

        ## use full bin integral to fix the en_cross 
        full, full_en_cross, full_branch,  _ = self.bin_integral(idx, BetaInverse, fix_en_cross=None)

        ## add s2integral over the partial bin 
        en_0, en_1 = en[0], ecut
        ri_0, ri_1 = ri[0], self.ri_(ecut)

        s2integral[-1], en_cross, branch, meta = self.s2_linear_integral( [en_0, en_1], [ri_0, ri_1], BetaInverse, fix_en_cross=full_en_cross )

        meta[-5] = s2integral[:-1].sum()
        meta[-4] = s2integral.sum()
        meta[-3] = float(idx)
        meta[-2] = ecut
        meta[-1] = float(line)

        if dump:
            print(" ecut %10.4f idx %3d  en %15s ri %15s  s2integral %s  " % (ecut, idx, str(en), str(ri), str(s2integral) ))     
        pass 
        return s2integral, meta


    def cs2_integral(self, efin, BetaInverse=1.5):
        """
        Slight non-monotonic all from the partial bin.

        Non-monotonic fixed by forcing use of common en_cross obtained from the full bin integral
        in the partial bin integrals. 
        """
        cs2i = np.zeros( [len(efin), len(self.ri)+1])
        meta = np.zeros( [len(efin), 12])

        for line,ecut in enumerate(efin):
            cs2i[line], meta[line] = self.ecut_integral(line, ecut, BetaInverse)
        pass  
        return cs2i, meta

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(edgeitems=50, precision=4)

    s2 = S2()
    s2.bin_dump() 

    s2.ecut_integral(0, 15.5, dump=True)

    nx = 100 
    efin = np.linspace( s2.ri[0,0], s2.ri[-1,0], nx )
    cs2i, meta = s2.cs2_integral(efin, BetaInverse=1.5)

    # np.c_[cs2i, np.arange(len(efin)), cs2i.sum(axis=1), efin  ]

    print(np.diff(cs2i.sum(axis=1)))

 
