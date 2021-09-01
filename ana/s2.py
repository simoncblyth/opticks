#/usr/bin/env python
"""



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

    def s2_linear_integral(self, en_, ri_, BetaInverse, fix_en_cross=None):
        """
        :param en_:
        :param ri_:
        :param BetaInverse:
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


             0.  
             |  .
             |     .
             +-------X----------
                        .
                           1


         Similar triangles to find the en_cross where ri == BetaInverse.
         This has advantage over s2 crossing zero in that it is more linear, 
         so the crossing should be a bit more precise. 


             en_cross - en_0            en_1 - en_0
             -------------------  =  --------------------
             BetaInverse - ri_0         ri_1 - ri_0 

             en_cross_0 =  en_0 + (BetaInverse - ri_0)*(en_1 - en_0)/(ri_1 - ri_0)


             en_1 - en_cross            en_1 - en_0
             -------------------  =  --------------------
             ri_1 - BetaInverse         ri_1 - ri_0 
  
             en_cross_1 =  en_1 - (ri_1 - BetaInverse)*(en_1 - en_0)/(ri_1 - ri_0)
             en_cross_1 =  en_1 + (BetaInverse - ri_1)*(en_1 - en_0)/(ri_1 - ri_0)


             ## when there is no crossing in the range en_cross will not be between en_0 and en_1
             ## note potential infinities for flat ri bins,  ri_1 == ri_0 

        """

        en = np.asarray(en_)
        ri = np.asarray(ri_)

        ct = BetaInverse/ri
        s2 = (1.-ct)*(1.+ct) 

        en_0, en_1 = en
        ri_0, ri_1 = ri 
        s2_0, s2_1 = s2

        branch = 0
        if s2_0 <= 0. and s2_1 <= 0.:      # s2 -ve OR 0.      : no CK in bin 
            branch = 1
        elif s2_0 < 0. and s2_1 > 0.:      # s2 goes +ve       : CK on RHS of bin
            branch = 2
        elif s2_0 >= 0. and s2_1 >= 0.:    # s2 +ve or zero    : CK across full bin
            branch = 3
        elif s2_0 > 0. and s2_1 <= 0.:     # s2 goes -ve or 0. : CK on LHS of bin
            branch = 4
        else:
            assert 0 
        pass
        
        if branch == 2 or branch == 4:     
            en_cross_A = (s2_1*en_0 - s2_0*en_1)/(s2_1 - s2_0)    ## s2 zeros should occur at ~same en_cross, but non-linear so less precision 
            en_cross_B = en_0 + (BetaInverse - ri_0)*(en_1 - en_0)/(ri_1 - ri_0)
            en_cross_C = en_1 + (BetaInverse - ri_1)*(en_1 - en_0)/(ri_1 - ri_0)
            en_cross = en_cross_B if fix_en_cross is None else fix_en_cross
        else:
            en_cross_A = np.nan
            en_cross_B = np.nan
            en_cross_C = np.nan
            en_cross = np.nan
        pass

        if branch == 1.: 
            area = 0.
        elif branch == 2:                            # s2 goes +ve 
            area = (en_1 - en_cross)*s2_1*0.5
        elif branch == 3:                            # s2 +ve or zero across full range 
            area = (en_1 - en_0)*(s2_0 + s2_1)*0.5
        elif branch == 4:                            # s2 goes -ve or to zero 
            area = (en_cross - en_0)*s2_0*0.5
        else:
            assert 0 
        pass

        Rfact = 369.81 / 10.
        s2i = Rfact * area
        
        meta = np.zeros(16)

        meta[0] = en_0
        meta[1] = en_1
        meta[2] = s2_0
        meta[3] = s2_1
    
        meta[4] = ri_0
        meta[5] = ri_1
        meta[6] = float(branch)
        meta[7] = s2i

        meta[8] = en_cross_A
        meta[9] = en_cross_B
        meta[10] = en_cross_C
        meta[11] = en_cross

        meta[12] = 0.
        meta[13] = 0.
        meta[14] = 0.
        meta[15] = 0.

        assert s2i >= 0.  
        return s2i, en_cross, branch, meta 
 
    def bin_integral(self, idx, BetaInverse, fix_en_cross=None): 
        """
        :param idx: 0-based bin index  
        :param BetaInverse: scalar
        :return s2integral: scalar sin^2 ck_angle integral in energy range  
        """
        en, ri = self.bin(idx) 
        return self.s2_linear_integral(en, ri, BetaInverse, fix_en_cross=fix_en_cross )

    def full_integral(self, BetaInverse): 
        """
        :param idx: 0-based bin index  
        :param BetaInverse: scalar
        :return s2integral: scalar sin^2 ck_angle integral in energy range  
        """
        s2integral = 0. 
        for idx in range(1, len(self.ri)):
            s2i, en_cross, branch, meta  = self.bin_integral(idx, BetaInverse)
            s2integral += s2i 
        pass
        return s2integral   

    def full_integral_scan(self):
        bis = np.linspace(1, 2, 11)
        for BetaInverse in bis:
            s2integral = self.full_integral(BetaInverse)
            print(" bi %7.4f s2integral  %7.4f  " % (BetaInverse, s2integral))
        pass

    def cut_integral(self, ecut, BetaInverse=1.5, dump=False, line=0 ):
        """
        """
        ## find index of the bin containing the ecut value  
        idx = int(np.digitize( ecut, self.ri[:,0], right=False ))

        s2integral = np.zeros(len(self.ri)+1)

        ## first add up integrals from full bins to the left of idx bin 
        for i in range(1,idx):  
            s2integral[i], _en_cross, _branch, _meta = self.bin_integral(i, BetaInverse)
        pass

        ## find details of the idx bin 
        en, ri = self.bin(idx)

        ## use full bin integral to obtain en_cross 
        full, full_en_cross, full_branch,  _ = self.bin_integral(idx, BetaInverse, fix_en_cross=None)

        ## add s2integral over the partial bin 
        en_0, en_1 = en[0], ecut
        ri_0, ri_1 = ri[0], self.ri_(ecut)

        # fix_en_cross only gets used when appropriate  
        s2integral[-1], en_cross, branch, meta = self.s2_linear_integral( [en_0, en_1], [ri_0, ri_1], BetaInverse, fix_en_cross=full_en_cross )

        tot = s2integral.sum()

        meta[-4] = tot
        meta[-3] = float(idx)
        meta[-2] = ecut
        meta[-1] = float(line)

        if dump:
            print("cut_integral  %10.4f idx %3d  en %15s ri %15s  tot %10.4f  " % (ecut, idx, str(en), str(ri), tot ))     
        pass 
        return s2integral, meta

    def cs2_integral(self, efin, BetaInverse=1.5):
        """
        Fixed slight non-monotonic by forcing use of common en_cross obtained from the full bin integral
        in the partial bin integrals.  Further improvement using ri-BetaInverse cross rather than s2 cross, 
        as thats more linear.
        """
        cs2i = np.zeros( [len(efin), len(self.ri)+1])
        meta = np.zeros( [len(efin), 16])

        for line,ecut in enumerate(efin):
            cs2i[line], meta[line] = self.cut_integral(ecut, BetaInverse, line=line)
        pass  
        s_cs2i = cs2i.sum(axis=1)  # sum over full bins and the partial bin for each cut 

        monotonic = np.all(np.diff(s_cs2i) >= 0.)
        assert monotonic
        return cs2i, meta


def test_full_vs_cut(s2, dump=False):
    bis = np.linspace(1., 2., 101)
    res = np.zeros( (len(bis), 4) )
    for i, BetaInverse in enumerate(bis): 
        full = s2.full_integral(BetaInverse)
        cut , _cut_meta = s2.cut_integral(s2.ri[-1,0], BetaInverse, dump=dump)
        cut_tot = cut.sum() 
        res[i] = [BetaInverse, full, cut_tot, np.abs(full-cut_tot)*1e10 ]
        assert np.abs(full - cut_tot) < 1e-10 
    pass
    return res 



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(edgeitems=50, precision=4)

    s2 = S2()

if 0:
    s2.bin_dump() 
    s2.full_integral_scan()
    res = test_full_vs_cut(s2) 

if 1:
    nx = 100 
    efin = np.linspace( s2.ri[0,0], s2.ri[-1,0], nx )
    cs2i, meta = s2.cs2_integral(efin, BetaInverse=1.5)

    # np.c_[cs2i, np.arange(len(efin)), cs2i.sum(axis=1), efin  ]

 
