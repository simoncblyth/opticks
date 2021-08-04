#!/usr/bin/env python
"""
rindex.py : creating a 2D Cerenkov ICDF lookup texture
========================================================

This demonstrates a possible approach for constructing a 2d Cerenkov ICDF texture::

    ( BetaInverse[nMin:nMax], u[0:1] )  

The result is being able to sample Cerenkov energies for photons radiated by a particle 
traversing the media with BetaInverse (between nMin and nMax of the material) 
with a single random throw followed by a single 2d texture lookup.
That compares with the usual rejection sampling approach which requires a variable number of throws 
(sometimes > 100) with double precision math needed to reproduce Geant4 results.

For BetaInverse 


"""
import os, numpy as np, logging
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt 
from scipy import optimize   
from opticks.ana.main import opticks_main 
from opticks.ana.key import keydir


def findbin(ee, e):
    """
    :param ee: array of ascending values 
    :param e: value to check 
    :return ie: index of first bin that contains e or -1 if not found

    A value equal to lower bin edge is regarded to be contained in the bin. 

    Hmm currently a value equal to the upper edge of the last bin 
    is not regarded as being in the bin::

        In [22]: findbin(ri[:,0], 15.5)
        Out[22]: -1

        In [23]: findbin(ri[:,0], 15.5-1e-6)
        Out[23]: 16


    Hmm could use binary search to do this quicker.
    """
    assert np.all(np.diff(ee) > 0)
    ie = -1
    for i in range(len(ee)-1):
        if ee[i] <= e and e < ee[i+1]: 
            ie = i
            break 
        pass 
    pass
    return ie    



def find_cross(ri, BetaInverse):
    """
    The interpolated rindex is piecewise linear 
    so can find the roots (where rindex crosses BetaInverse)
    without using optimization : just observing sign changes
    to find crossing bins and then some linear calc::
                 
                  (x1,v1)
                   *
                  / 
                 /
                /
        -------?(x,v)----    v = 0    when values are "ri_ - BetaInverse"
              /
             /
            /
           /
          * 
        (x0,v0)      


         Only x is unknown 


              v1 - v        v - v0
             ----------  =  ----------
              x1 - x        x - x0  


           v1 (x - x0 ) =  -v0  (x1 - x )

           v1.x - v1.x0 = - v0.x1 + v0.x  

           ( v1 - v0 ) x = v1*x0 - v0*x1


                         v1*x0 - v0*x1
               x    =   -----------------
                          ( v1 - v0 ) 


    Hmm with "v0*v1 < 0" this is failing to find a crossing at a bin edge::

        find_cross(ckr.ri, 1.4536)  = []

    ::

                                                     +----
                                                    / 
           ----+                                   /
                \                                 /
                 \                               /
         . . . . .+----------------+------------+ . . . .
                  ^                ^
              initially          causes nan 
              not found          without v0 != v1 protection

    With "v0*v1 <= 0 and v0 != v1" it manages to find the crossing.
    Need protection against equal v0 and v1 to avoid nan from 
    mid or endpoint of "tangent" plateaus::

        find_cross(ckr.ri, 1.4536)  = array([10.33])

    """
    cross = []
    for i in range(len(ri)-1):
        x0 = ri[i,0]
        x1 = ri[i+1,0]
        v0 = BetaInverse - ri[i,1]
        v1 = BetaInverse - ri[i+1,1]

        if v0*v1 <= 0 and v0 != v1:
            x = (v1*x0 - v0*x1)/(v1-v0)     
            #print("find_cross i %d x0 %6.4f x1 %6.4f v0 %6.4f v1 %6.4f x %6.4f " % (i, x0,x1,v0,v1,x))
            cross.append(x)
            pass
        pass   
    pass
    return np.array(cross) 


class CKRindex(object):
    """
    Cerenkov Sampling cases:

    * BetaInverse between 1 and nMin there are no crossings and Cerenkov is permitted across the full domain
    * BetaInverse  > nMax there are no crossins and Cerenkov is not-permitted across the full domain
    * BetaInverse between nMin and nMax there will be crossings and permitted/non-permitted regions depending on rindex 

    """
    def __init__(self):
        kd = keydir(os.environ["OPTICKS_KEY"])
        ri = np.load(os.path.join(kd, "GScintillatorLib/LS_ori/RINDEX.npy"))
        ri[:,0] *= 1e6  
        ri_ = lambda e:np.interp(e, ri[:,0], ri[:,1] )
        nMax = ri[:,1].max() 
        nMin = ri[:,1].min() 

        self.ri = ri
        self.ri_ = ri_
        self.nMax = nMax
        self.nMin = nMin


    def find_energy_range(self, bis):
        log.info("find_energy_range")
        ri = self.ri 
        xri = {} 
        xrg = {}  # dict of range arrays with min and max permissable energies

        for bi in bis:
            xri[bi] = find_cross(ri, BetaInverse=bi)   # list of crossings 

            lhs = ri[0,1] - bi > 0   # ck allowed at left edge of domain ?
            rhs = ri[-1,1] - bi > 0  # ck allowed at right edge of domain ?  

            if len(xri[bi]) > 1:
                xrg[bi] = np.array( [xri[bi].min(), xri[bi].max()])
            elif len(xri[bi]) == 1:
                # one crossing needs special handling as need to define an energy range with one side or the other
                if lhs and not rhs:
                    xrg[bi] = np.array([ ri[0,0], xri[bi][0] ])
                elif not lhs and rhs:   
                    xrg[bi] = np.array([ xri[bi][0], ri[-1,0] ])
                else:
                    log.fatal("unexpected 1-crossing")
                    assert 0 
                pass
                #log.info("bi %s one crossing : lhs %s rhs %s xrg[bi] %s  " % (bi, lhs, rhs, str(xrg[bi]) ))
            elif len(xri[bi]) == 0:
                if lhs and rhs:
                    xrg[bi] = np.array( [ ri[0,0], ri[-1,0] ])
                else:
                    xrg[bi] = None
                pass
                #log.info("bi %s zero crossing : lhs %s rhs %s xrg[bi] %s  " % (bi, lhs, rhs, str(xrg[bi]) ))
            else:
                xrg[bi] = None
            pass
        pass
        #log.info("xri\n%s", xri)
        #log.info("xrg\n%s", xrg)
        self.xri = xri
        self.xrg = xrg
    pass

    def find_energy_range_plot(self, bis):
        ri = self.ri 
        xri = self.xri
        title = "rindex.py : ind_energy_range_plot"

        fig, ax = plt.subplots(figsize=ok.figsize); 
        fig.suptitle(title)

        # steps make no sense for rindex, as it is inherently interpolated between measured points
        ax.plot( ri[:,0], ri[:,1] )  

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.plot( xlim, [self.nMax,self.nMax], label="nMax", linestyle="dotted", color="r" )
        ax.plot( xlim, [self.nMin,self.nMin], label="nMin", linestyle="dotted", color="r" )

        for bi in bis:
            ax.plot( xlim, [bi,bi], linestyle="dotted", color="r" )
        pass  
        ax.plot( [ri[0,0], ri[0,0]], ylim, linestyle="dotted", color="r" )
        ax.plot( [ri[-1,0], ri[-1,0]], ylim, linestyle="dotted", color="r" )

        for bi in bis:
            for x in xri[bi]:
                ax.plot( [x,x], ylim, linestyle="dotted", color="r" )
            pass
        pass
        ax.scatter( ri[:,0], ri[:,1] )
        fig.show()


    def s2_cumsum(self, bis):
        """
        This uses a lot of energy bins across the allowed Cerenkov range 
        so probably no need for careful bin crossing the allowed edges ?
        """
        log.info("s2_cumsum")
        xrg = self.xrg
        ed = {}     # dict of energy domain arrays
        s2e = {}    # dict of s2 arrays across the energy domain
        cs2e = {}   # dict of cumumlative sums across the energy domain

        ri = self.ri  
        ## np.minimum prevents the cos(th) from exceeding 1 in disallowed regions
        ct_ = lambda bi,e:np.minimum(1.,bi/np.interp(e, ri[:,0], ri[:,1] ))
        s2_ = lambda bi,e:(1-ct_(bi,e))*(1+ct_(bi,e))

        for bi in bis:
            if xrg[bi] is None: continue
            ed[bi] = np.linspace(xrg[bi][0],xrg[bi][1],4096)    # energy range from min to max allowable
            s2e[bi] = s2_(bi,ed[bi])  
            cs2e[bi] = np.cumsum(s2e[bi])  

            if cs2e[bi][-1] > 0.:
                cs2e[bi] /= cs2e[bi][-1]      # last bin will inevitably be maximum one as cumulative   
            else:
                log.fatal("bi %7.4f zero cannot normalize " % bi )
            pass
        pass 
        self.ed = ed  
        self.s2e = s2e  
        self.cs2e = cs2e  

    def s2_cumsum_plot(self, bis):
        xrg = self.xrg
        ed = self.ed 
        cs2e = self.cs2e
        title = "rindex.py : s2_cumsum_plot"

        fig, axs = plt.subplots(figsize=ok.figsize) 
        fig.suptitle(title)
        for i, bi in enumerate(bis):
            if xrg[bi] is None: continue
            ax = axs if axs.__class__.__name__ == 'AxesSubplot' else axs[i] 
            ax.plot( ed[bi], cs2e[bi], label="cs2e : integrated s2 vs e bi:%6.4f  " % bi )
            #ax.set_xlim( xrg[1.5][0], xrg[1.5][1] )
            ax.set_ylim( -0.1, 1.1 ) 
            ax.legend()
 
        pass
        fig.show()

    def s2_integrate__(self, BetaInverse, en_0, en_1, ri_0, ri_1 ):
        """
        :param BetaInverse: 
        :param en_0: 
        :param en_1: 
        :param ri_0: 
        :param ri_1: 
        :return ret: scalar integrated value for the bin

        When s2 is positive across the bin this returns the trapezoidal area.

        When there is an s2 zero crossing a linear calculation is used to find 
        the crossing point and the triangle area is returned that excludes negative contributions. 
        """
        ct_0 = BetaInverse/ri_0
        ct_1 = BetaInverse/ri_1

        s2_0 = (1.-ct_0)*(1.+ct_0) 
        s2_1 = (1.-ct_1)*(1.+ct_1) 

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
            print( "s2_integrate__  en_0 %10.5f ri_0 %10.5f s2_0 %10.5f  en_1 %10.5f ri_1 %10.5f s2_1 %10.5f " % (en_0, ri_0, s2_0, en_1, ri_1, s2_1 )) 
            assert 0 
        pass
        assert ret >= 0. 
        return ret


    def s2_integrate_(self, BetaInverse, edom):
        """
        :param BetaInverse: scalar
        :param edom: array of energy cuts
        :return s2ij: array of the same length as edom containing the integral values up to the edom energy cuts 

        Note that s2in should be strictly monotonic, never decreasing, from bin to bin
        Bin before last somehow manages to break that for BetaInverse 1.4536::

             np.c_[ckr.edom[1.4536], ckr.s2ij[1.4536]][2000:2100] 
             np.c_[ckr.edom[1.4536], ckr.s2ij[1.4536]][4000:]    

        Possible dispositions of the ecut with respect to the bin: 
                                        
                    en0_b    en1_b
 
                      |        |     [1]      en0_b < ecut and en1_b < ecut 
                      |        |    
                      |  [2]   |              en0_b < ecut and ecut < en1_b     
                      |        |    
                 [3]  |        |              bin does not contibute    
                      |        |    

        """
        ri_ = self.ri_
        ni = len(edom)
        nj = len(self.ri)-1
        s2ij = np.zeros( (ni,nj), dtype=edom.dtype)

        # for each energy cut, sum contributions from each rindex bin
        for i in range(ni):        
            en_cut = edom[i]  
            ri_cut = ri_(en_cut)

            for j in range(nj):
                en0_b = self.ri[j,0]
                en1_b = self.ri[j+1,0]

                ri0_b = self.ri[j,1]
                ri1_b = self.ri[j+1,1]

                allowed = True
                if en0_b < en_cut and en1_b <= en_cut:                          # full bin included in cumulative range     
                    en_0, en_1 = en0_b, en1_b  
                    ri_0, ri_1 = ri0_b, ri1_b 
                elif en0_b <= en_cut and en_cut <= en1_b:                        #  en0_b < ecut < en1_b :  ecut divides the bin 
                    en_0, en_1 = en0_b, en_cut 
                    ri_0, ri_1 = ri0_b, ri_cut
                else:
                    allowed = False
                pass
                if allowed:
                    s2ij[i,j] = self.s2_integrate__( BetaInverse, en_0, en_1, ri_0, ri_1 ) 
                pass
            pass    
        pass
        return s2ij


    def s2_integrate(self, bis, nx=4096):
        """
        * CONCLUSION : USE S2SLIVER NOT THIS 

        This approach is slow and has a problem of 
        giving slightly non-monotonic CDF for the pathalogical BetaInverse=nMin. 
        As a result of this implemented s2sliver_integrate which is much 
        faster and avoids the problem

        This was aiming to be more accurate that the original quick 
        and dirty s2_cumsum but in retrospect the s2sliver_integrate 
        approach is much better combining better accuracy, speed
        and simplicity.

        Follows approach of ckn.py:GetAverageNumberOfPhotons_s2  
        although it turns out to not be simpler that np.cumsum approach.

        Need cumulative s2 integral.
        For each BetaInverse need to compute the s2 integral over
        an increasing energy range that will progressively 
        cover more and more rindex bins until eventually 
        covering them all.

        Notice no need to compute permissable ranges as the 
        sign of s2 shows that with no further effort.

        HMM the "cdf" is going above 1 for the top bin for BetaInverse below nMin 
        where CK can happen across entire range. 

        * That was a "<" which on changing to "<=" avoided the problem of not properly 
          accounting for the last bin.

        Also when BetaInverse = 1.4536 = ckr.ri[:,1].min() that 
        exactly matches the rindex of the last two energy points::

               ...
               [ 9.538 ,  1.5545],
               [10.33  ,  1.4536],
               [15.5   ,  1.4536]])

        so the ct = 1 and s2 = 0 across the entire top bin. 
        That somehow causes mis-normalization for full-range CK ?

        So there is a discrepancy in handling of the top bin between the np.cumsum 
        and the trapezoidal integral approaches.

        Possibly an off-by-one with the handling of the edom/ecut ?

        Sort of yes, actually most of the issue seems due to using "< ecut" and not "<= ecut" 
        which resulted in s2in not including the top bin causing it to be non-monotonic and 
        resultued in the normalization being off.

        Hmm that is an advantage with using cumsum rather than lots of integrals
        with increasing range.

        Getting a small decrement with increasing ecut in range 10.30-10.33 for the probematic 1.4536::

            In [6]: np.c_[np.diff(ckr.s2in[1.4536])[-20:]*1e6,ckr.s2in[1.4536][-20:],ckr.edom[1.4536][-20:]]                                                                                                
            Out[6]: 
            array([[  3.0319,   1.2293,  10.2893],
                   [  2.1562,   1.2293,  10.2914],
                   [  1.2797,   1.2293,  10.2936],
                   [  0.4026,   1.2293,  10.2957],
                   [ -0.4753,   1.2293,  10.2978],
                   [ -1.3539,   1.2293,  10.3   ],
                   [ -2.2333,   1.2293,  10.3021],
                   [ -3.1134,   1.2293,  10.3043],
                   [ -3.9942,   1.2293,  10.3064],
                   [ -4.8758,   1.2293,  10.3086],
                   [ -5.7581,   1.2293,  10.3107],
                   [ -6.6411,   1.2293,  10.3128],
                   [ -7.5249,   1.2293,  10.315 ],
                   [ -8.4094,   1.2293,  10.3171],
                   [ -9.2947,   1.2293,  10.3193],
                   [-10.1807,   1.2293,  10.3214],
                   [-11.0674,   1.2293,  10.3236],
                   [-11.9549,   1.2293,  10.3257],
                   [-12.8431,   1.2292,  10.3279],
                   [-13.7321,   1.2292,  10.33  ]])

            In [7]:                                                             

        """
        log.info("s2_integrate")
        ri = self.ri

        s2ij = {}
        s2in = {}
        cs2in = {}
        edom = {}
        yrg = {}

        for BetaInverse in bis:
            numPhotons_s2, emin, emax = self.GetAverageNumberOfPhotons_s2(BetaInverse)
            yrg[BetaInverse] = [numPhotons_s2, emin, emax]
            if numPhotons_s2 <= 0.: continue
            edom[BetaInverse] = np.linspace(emin, emax, nx) 
            s2ij[BetaInverse] = self.s2_integrate_(BetaInverse, edom[BetaInverse])
            s2in[BetaInverse] = s2ij[BetaInverse].sum(axis=1)
            cs2in[BetaInverse] = s2in[BetaInverse]/s2in[BetaInverse][-1]   # normalization to last bin
            d = np.diff(s2in[BetaInverse])
            if len(d[d<0]) > 0:
                log.fatal(" monotonic diff check fails for BetaInverse %s  d[d<0] %s " % (BetaInverse, str(d[d<0])))   
                print(d)
            pass
        pass

        self.s2ij = s2ij
        self.s2in = s2in
        self.cs2in = cs2in
        self.edom = edom
        self.yrg = yrg

    def s2_integrate_plot(self, bis):
        cs2in = self.cs2in
        edom = self.edom
        title = "rindex.py : s2_integrate_plot"

        fig, axs = plt.subplots(figsize=ok.figsize) 
        fig.suptitle(title)
        for i, bi in enumerate(bis):
            if not bi in cs2in: continue
            ax = axs if axs.__class__.__name__ == 'AxesSubplot' else axs[i] 
            ax.plot( edom[bi], cs2in[bi], label="cs2in : integrated s2 vs e bi:%6.4f  " % bi )
            ax.set_ylim( -0.1, 1.1 )  
            ax.legend()
        pass
        fig.show()

    def s2sliver_check(self, edom):
        """
        seems not needed
        """
        ri = self.ri
        eb = ri[:,0]
        contain = 0 
        straddle = 0 

        for i in range(len(edom)-1):

            en_0 = edom[i]
            en_1 = edom[i+1]

            ## find bin indices corresponding to left and right sliver edges
            b0 = findbin(eb, en_0)  
            b1 = findbin(eb, en_1)

            found =  b0 > -1 and b1 > -1
            if not found:
                log.fatal(" bin not found %s %s %d %d " % (en_0,en_1,b0,b1 )) 
            pass
            assert(found)

            if b0 == b1:
                # sliver is contained in single rindex bin
                pass
                contain += 1 
            elif b1 == b0 + 1:
                # sliver straddles bin edge 
                pass
                straddle += 1 
            else:
                log.fatal("unexpected bin disposition en_0 %s en_1 %s b0 %d b1 %d " % (en_0,en_1,b0,b1)) 
                assert 0
            pass
        pass
        log.info("contain %d straddle %d sum %d " % (contain, straddle, contain+straddle))


    def s2sliver_integrate_(self, BetaInverse, edom):
        """
        :param BetaInverse: scalar
        :param edom: array of energy values in eV 
        :return s2slv: array with same shape as edom containing s2 definite integral values within energy slivers 


        Possible dispositions of the energy sliver with respect to the bin: 
                                        
        1. sliver fully inside a bin:: 

 
                      |  .   .    |                       |
                      |  .   .    |                       | 
                      |  .   .    |                       |  
                      |  .   .    |                       | 

 
        2. sliver straddles edge of bin::
 
                      |          . | .                    |   
                      |          . | .                    |
                      |          . | .                    |
                      |          . | .                    |


        Within the sliver there is the possibility of s2 crossings
        that will require constriction of the sliver, this is handled 
        by s2_integrate__
        """
        ri_ = self.ri_
        s2slv = np.zeros( (len(edom)), dtype=edom.dtype )
        for i in range(len(edom)-1):
            en_0 = edom[i]
            en_1 = edom[i+1]
            ri_0 = ri_(en_0) 
            ri_1 = ri_(en_1) 
            s2slv[i+1] = self.s2_integrate__( BetaInverse, en_0, en_1, ri_0, ri_1 )  
            # s2_integrate__ accounts for s2 crossings : "chopping bins" and giving triangle areas 
        pass
        return s2slv 


    def s2sliver_integrate(self, bis, nx=4096):
        """
        Issues with small breakage of monotonic, makes me think its better to structure the calculation 
        to make monotonic inevitable, by storing "sliver" integrals and np.cumsum adding them up
        to give the cumulative CDF.

        Do this by slicing the energy range for each BetaInverse into small "slivers" 
        Can assume that the sliver size is smaller than smallest rindex energy bin size, 
        so the sliver can either be fully inside the bin or straddling the bin edge.
        Then get the total integral by np.cumsum adding the pieces.
        This will be much more efficient too as just does 
        the integral over each energy sliver once. 
        """
        log.info("s2sliver_integrate")

        vdom = {}
        vrg = {}
        s2slv = {}
        cs2slv = {}

        for BetaInverse in bis:
            numPhotons_s2, emin, emax = self.GetAverageNumberOfPhotons_s2(BetaInverse)
            vrg[BetaInverse] = [numPhotons_s2, emin, emax]
            if numPhotons_s2 <= 0.: continue
            vdom[BetaInverse] = np.linspace(emin, emax, nx) 
            s2slv[BetaInverse] = self.s2sliver_integrate_(BetaInverse, vdom[BetaInverse])
            cs2slv[BetaInverse] = np.cumsum(s2slv[BetaInverse])
            cs2slv[BetaInverse] /= cs2slv[BetaInverse][-1]      # CDF normalize
        pass
        self.vrg = vrg
        self.vdom = vdom
        self.s2slv = s2slv
        self.cs2slv = cs2slv

    def s2sliver_integrate_plot(self, bis):
        cs2slv = self.cs2slv
        vdom = self.vdom
        title = "rindex.py : s2sliver_integrate_plot"

        fig, axs = plt.subplots(figsize=ok.figsize) 
        fig.suptitle(title)
        for i, bi in enumerate(bis):
            if not bi in cs2slv: continue
            ax = axs if axs.__class__.__name__ == 'AxesSubplot' else axs[i] 
            ax.plot( vdom[bi], cs2slv[bi], label="cs2slv : cumsum of s2 sliver integrals,  bi:%6.4f  " % bi )
            #ax.set_ylim( -0.1, 1.1 )  
            ax.legend()
        pass
        fig.show()


    def comparison_plot(self, bis):

        cs2in = self.cs2in if hasattr(self, 'cs2in') else None
        edom = self.edom if hasattr(self, 'edom') else None
        yrg = self.yrg if hasattr(self, 'yrg') else None

        cs2slv = self.cs2slv
        vdom = self.vdom

        ri = self.ri  
        xrg = self.xrg
        ed = self.ed 
        cs2e = self.cs2e

        titls = ["rindex.py : comparison_plot %s " % str(bis), ]

        bi = bis[0] if len(bis) == 1 else None
        if len(bis) == 1:
            if not xrg is None and bi in xrg: 
                titls.append(" xrg[bi] %s " % str(xrg[bi]))
            pass
            if not yrg is None and bi in yrg: 
                titls.append(" yrg[bi] %s " % str(yrg[bi]))
            pass
            if not edom is None and bi in edom:
                titls.append(" edom[bi] %s " % str(edom[bi]))
            pass
        pass
        title = "\n".join(titls) 

        fig, axs = plt.subplots(figsize=ok.figsize) 
        fig.suptitle(title)

        for i, bi in enumerate(bis):
            if xrg[bi] is None: continue

            if not cs2in is None: 
                if not bi in cs2in: continue
            pass
            if not bi in vdom: continue


            ax = axs if axs.__class__.__name__ == 'AxesSubplot' else axs[i] 
            ax.plot( ed[bi], cs2e[bi], label="cs2e : integrated s2 vs e bi:%6.4f  " % bi )

            if not edom is None and not cs2in is None:
                ax.plot( edom[bi], cs2in[bi], label="cs2in : integrated s2 vs e bi:%6.4f  " % bi )
            pass

            ax.plot( vdom[bi], cs2slv[bi], label="cs2slv : cumsum of s2 sliver integrals vs e bi:%6.4f  " % bi )

            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            ax.plot( xlim, [1., 1.], label="one", linestyle="dotted", color="r" )

            for e in ri[:,0]:
                ax.plot( [e, e], ylim, linestyle="dotted", color="r" )
            pass

            ax.legend()
            #ax.set_ylim( 0, 1 )   # huh getting > 1 on rhs ?
        pass
        fig.show()


    def GetAverageNumberOfPhotons_s2(self, BetaInverse, charge=1, dump=False ):
        """
        see ana/ckn.py for development of this

        Simplfied Alternative to _s2messy following C++ implementation. 
        Allowed regions are identified by s2 being positive avoiding the need for 
        separately getting crossings. Instead get the crossings and do the trapezoidal 
        numerical integration in one pass, improving simplicity and accuracy.  
    
        See opticks/examples/Geant4/CerenkovStandalone/G4Cerenkov_modified.cc
        """
        s2integral = 0.
        emin = self.ri[-1, 0]  # start at maximum
        emax = self.ri[0, 0]   # start at minimum 

        for j in range(len(self.ri)-1):

            en0_b = self.ri[j,0]
            en1_b = self.ri[j+1,0]

            ri0_b = self.ri[j,1]
            ri1_b = self.ri[j+1,1]

            en = np.array([en0_b, en1_b ]) 
            ri = np.array([ri0_b, ri1_b ]) 

            ct = BetaInverse/ri
            s2 = (1.-ct)*(1.+ct) 

            ## The en and s2 start off corresponding to full bin.
            ## When s2 crossings happen within the bin the en and s2 are adjusted 
            ## for the crossing point so can add edge triangles to body trapezoids.

            if s2[0] <= 0. and s2[1] <= 0.:     # Cerenkov not permissable in this bin 
                en = None                        
            elif s2[0] < 0. and s2[1] > 0.:     # Cerenkov can happen at high energy side of bin
                en[0] = (s2[1]*en[0] - s2[0]*en[1])/(s2[1] - s2[0])
                s2[0] = 0.
            elif s2[0] >= 0. and s2[1] >= 0.:   # Cerenkov can happen across entire bin 
                pass
            elif s2[0] > 0. and s2[1] < 0.:     # Cerenkov can happen at low energy side of bin 
                en[1] = (s2[1]*en[0] - s2[0]*en[1])/(s2[1] - s2[0]) 
                s2[1] = 0. 
            else:                               # unhandled situation  
                en = None
                print( " en_0 %10.5f ri_0 %10.5f s2_0 %10.5f  en_1 %10.5f ri_1 %10.5f s2_1 %10.5f " % (en[0], ri[0], s2[0], en[1], ri[1], s2[1] )) 
                assert 0 
            pass

            if dump:
                print( " j %2d en_0 %10.5f ri_0 %10.5f s2_0 %10.5f  en_1 %10.5f ri_1 %10.5f s2_1 %10.5f " % (j, en[0], ri[0], s2[0], en[1], ri[1], s2[1] )) 
            pass

            if not en is None:
                emin = min(en[0], emin)
                emax = max(en[1], emax)
                s2integral +=  (en[1] - en[0])*(s2[0] + s2[1])*0.5    # tapezoidal integration 
            pass
        pass
        Rfact = 369.81 / 10. #        Geant4 mm=1 cm=10    
        NumPhotons = Rfact * charge * charge * s2integral
        return NumPhotons, emin, emax 


    def make_lookup_samples(self, bis):
        """
        After viewing some rejection sampling youtube vids realise 
        that must use s2 (sin^2(th) in order to correspond to the pdf that the 
        rejection sampling is using. Doing that is giving a good match (chi2/ndf 1.1-1.2) 
        using numerical integration and inversion with 4096 bins.

        So this becomes a real cheap (and constant) way to sample from the Cerenkov energy distrib.  
        But the problem is the BetaInverse input.

        Presumably would need to use 2d texture lookup to bake lots of different ICDF 
        for different BetaInverse values.

        Although the "cutting" effect of BetaInverse on the CDF is tantalising, 
        it seems difficult to use.
        """
        log.info("make_lookup_samples")
        xrg = self.xrg
        cs2e = self.cs2e
        ed = self.ed


        ## THIS IS THE CRUCIAL INVERSION OF CDF : 
        ## FORMING A FUNCTION THAT TAKES RANDOM u(normalized CDF value)  AND BetaInverse 
        ## AS INPUT AND RETURNS AN ENERGY SAMPLE
        ## IMPLEMENTED VIA LINEAR INTERPOLATION OF THE RELEVANT CDF FOR THE BetaInverse
        ## 
        look_ = lambda bi,u:np.interp(u, cs2e[bi], ed[bi] )

        l = {}
        for bi in bis:
            if xrg[bi] is None: continue
            u = np.random.rand(1000000)   
            l[bi] = look_(bi,u) 
        pass
        self.l = l 

    def save_lookup_samples(self, bis):
        log.info("save_lookup_samples")
        xrg = self.xrg
        l = self.l
        fold = "/tmp/rindex" 
        for i,bi in enumerate(bis):
            if xrg[bi] is None: continue
            path = os.path.join(fold,"en_integrated_lookup_1M_%d.npy" % i ) 
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            pass
            print("save to %s " % path)
            np.save(path, l[bi] ) 
        pass

    def make_lookup_samples_plot(self, bis):
        xrg = self.xrg
        l = self.l
        title = "rindex.py : make_lookup_samples_plot"

        fig, axs = plt.subplots(figsize=ok.figsize)
        fig.suptitle(title)
        for i, bi in enumerate(bis):
            if xrg[bi] is None: continue
            ax = axs if axs.__class__.__name__ == 'AxesSubplot' else axs[i] 
            hd = np.arange(xrg[bi][0],xrg[bi][1],0.1)   
            h = np.histogram(l[bi], hd )
            ax.plot( h[1][:-1], h[0][0:], drawstyle="steps-post", label="bi %6.4f " % bi)  
            ax.legend()
        pass
        fig.show()


    def load_QCerenkov_s2slv(self):
        """
        see also ckn.py for comparison with the numPhotons 
        """
        path = "/tmp/blyth/opticks/QCerenkovTest/test_getS2SliverIntegrals_many.npy"
        log.info("load %s " % path )
        s2slv = np.load(path) if os.path.exists(path) else None
        self.s2slv = s2slv 
        globals()["ss"] = s2slv

if __name__ == '__main__':

    plt.ion()
    ok = opticks_main()

    ckr = CKRindex()
    #bis = np.array(  [1.5,1.6,1.7] )
    #bis = np.array(  [1.6] )
    #bis = np.array(  [1.457] )   ## with one crossing, need to form a range with one side or the other depending on rindex at edges


    #bis = np.linspace(1.,ckr.nMin,10)
    bis = np.linspace(ckr.nMin,ckr.nMax,10)
    sbis = bis[6:7]
    #sbis = bis[-1:]



if 1:
    ckr.find_energy_range(bis)
    ckr.find_energy_range_plot(bis)

    ckr.s2_cumsum(bis)
    ckr.s2_cumsum_plot(bis)

    ckr.make_lookup_samples(bis)
    #ckr.save_lookup_samples(bis)        # allows chi2 comparison using ana/wavelength_cfplot.py 
    ckr.make_lookup_samples_plot(bis)

    #ckr.s2_integrate(bis)
    #ckr.s2_integrate_plot(bis)


if 1:
    ckr.s2sliver_integrate(bis)
    ckr.s2sliver_integrate_plot(bis)

    ckr.comparison_plot(sbis)


    ckr.load_QCerenkov_s2slv()


