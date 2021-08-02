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
    """
    cross = []
    for i in range(len(ri)-1):
        x0 = ri[i,0]
        x1 = ri[i+1,0]
        v0 = BetaInverse - ri[i,1]
        v1 = BetaInverse - ri[i+1,1]
        if v0*v1 < 0:
            x = (v1*x0 - v0*x1)/(v1-v0)     
            print("i %d x0 %6.4f x1 %6.4f v0 %6.4f v1 %6.4f x %6.4f " % (i, x0,x1,v0,v1,x))
            cross.append(x)
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
                log.info("bi %s one crossing : lhs %s rhs %s xrg[bi] %s  " % (bi, lhs, rhs, str(xrg[bi]) ))
            elif len(xri[bi]) == 0:
                if lhs and rhs:
                    xrg[bi] = np.array( [ ri[0,0], ri[-1,0] ])
                else:
                    xrg[bi] = None
                pass
                log.info("bi %s zero crossing : lhs %s rhs %s xrg[bi] %s  " % (bi, lhs, rhs, str(xrg[bi]) ))
            else:
                xrg[bi] = None
            pass
        pass
        print(xri)
        print(xrg)
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
            cs2e[bi] /= cs2e[bi][-1]      # last bin will inevitably be maximum one as cumulative   
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
            ax.plot( ed[bi], cs2e[bi], label="cs2e : integrated s2 vs e bi:%6.3f  " % bi )
            #ax.set_xlim( xrg[1.5][0], xrg[1.5][1] )
            ax.set_ylim( -0.1, 1.1 ) 
            ax.legend()
 
        pass
        fig.show()

    def s2_integrate__(self, BetaInverse, en, ri):
        """
        :param BetaInverse: scalar 
        :param en: array of 2 values
        :param ri: array of 2 values
        :return s2i: scalar integrated value for the bin
        """
        ct = BetaInverse/ri
        s2 = (1.-ct)*(1.+ct) 

        if s2[0] <= 0. and s2[1] <= 0.:
            return 0.
        elif s2[0] < 0. and s2[1] > 0.:
            en_cross = (s2[1]*en[0] - s2[0]*en[1])/(s2[1] - s2[0])
            s2_cross = 0.
            return  (en[1] - en_cross)*(s2_cross + s2[1])*0.5
        elif s2[0] >= 0. and s2[1] >= 0.:
            return (en[1] - en[0])*(s2[0] + s2[1])*0.5
        elif s2[0] > 0. and s2[1] < 0.:
            en_cross = (s2[1]*en[0] - s2[0]*en[1])/(s2[1] - s2[0]) 
            s2_cross = 0. 
            return  (en_cross - en[0])*(s2_cross + s2[0])*0.5
        else:
            print( " en_0 %10.5f ri_0 %10.5f s2_0 %10.5f  en_1 %10.5f ri_1 %10.5f s2_1 %10.5f " % (en[0], ri[0], s2[0], en[1], ri[1], s2[1] )) 
            assert 0 
        pass
        return 0. 


    def s2_integrate_(self, BetaInverse, edom):
        """
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
        s2in = np.zeros(len(edom), dtype=edom.dtype)

        for i in range(len(edom)):        
            en_cut = edom[i]  
            ri_cut = ri_(en_cut)

            for j in range(len(self.ri)-1):
                en0_b = self.ri[j,0]
                en1_b = self.ri[j+1,0]

                ri0_b = self.ri[j,1]
                ri1_b = self.ri[j+1,1]

                if en0_b < en_cut and en1_b < en_cut:                          # full bin included in cumulative range     
                    en = np.array([en0_b, en1_b ]) 
                    ri = np.array([ri0_b, ri1_b ]) 
                    s2in[i] += self.s2_integrate__( BetaInverse, en, ri ) 
                elif en0_b < en_cut and en_cut < en1_b:                        #  en0_b < ecut < en1_b :  ecut divides the bin 
                    en = np.array([en0_b, en_cut]) 
                    ri = np.array([ri0_b, ri_cut])
                    s2in[i] += self.s2_integrate__( BetaInverse, en, ri ) 
                else:
                    pass
                pass
            pass    
        pass
        cs2in = s2in
        cs2in /= cs2in[-1]
        return cs2in


    def s2_integrate(self, bis, nx=4096):
        """
        Follows approach of ckn.py:GetAverageNumberOfPhotons_s2  
        to provide a cleaner and simpler one pass implementation 

        Need cumulative s2 integral.
        For each BetaInverse need to compute the s2 integral over
        an increasing energy range that will progressively 
        cover more and more rindex bins until eventually 
        covering them all.

        Notice no need to compute permissable ranges as the 
        sign of s2 shows that with no further effort.

        HMM there is a binning resolution disadvantage with using the  
        the same range 
        """
        ri = self.ri
        cs2in = {}
        edom = {}
        yrg = {}

        for BetaInverse in bis:
            numPhotons_s2, emin, emax = self.GetAverageNumberOfPhotons_s2(BetaInverse)
            yrg[BetaInverse] = [numPhotons_s2, emin, emax]
            if numPhotons_s2 > 0.:
                edom[BetaInverse] = np.linspace(emin, emax, nx) 
                cs2in[BetaInverse] = self.s2_integrate_(BetaInverse, edom[BetaInverse])
            pass
        pass
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
            ax.plot( edom[bi], cs2in[bi], label="cs2in : integrated s2 vs e bi:%6.3f  " % bi )
            ax.set_ylim( -0.1, 1.1 )   # TODO: investigate getting > 1
            ax.legend()
        pass
        fig.show()

    def comparison_plot(self, bis):

        cs2in = self.cs2in
        edom = self.edom

        yrg = self.yrg
        xrg = self.xrg
        ed = self.ed 
        cs2e = self.cs2e

        titls = ["rindex.py : comparison_plot %s " % str(bis), ]

        bi = bis[0] if len(bis) == 1 else None
        if len(bis) == 1:
            titls.append(" xrg[bi] %s " % str(xrg[bi]))
            titls.append(" yrg[bi] %s " % str(yrg[bi]))
            titls.append(" edom[bi] %s " % str(edom[bi]))
        pass
        title = "\n".join(titls) 

        fig, axs = plt.subplots(figsize=ok.figsize) 
        fig.suptitle(title)

        for i, bi in enumerate(bis):
            if xrg[bi] is None: continue
            if not bi in cs2in: continue

            ax = axs if axs.__class__.__name__ == 'AxesSubplot' else axs[i] 
            ax.plot( ed[bi], cs2e[bi], label="cs2e : integrated s2 vs e bi:%6.3f  " % bi )
            ax.plot( edom[bi], cs2in[bi], label="cs2in : integrated s2 vs e bi:%6.3f  " % bi )

            xlim = ax.get_xlim()
            ax.plot( xlim, [1., 1.], label="one", linestyle="dotted", color="r" )

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
        emin = self.ri[-1, 0]
        emax = self.ri[0, 0]

        for j in range(len(self.ri)-1):

            en0_b = self.ri[j,0]
            en1_b = self.ri[j+1,0]

            ri0_b = self.ri[j,1]
            ri1_b = self.ri[j+1,1]

            en = np.array([en0_b, en1_b ]) 
            ri = np.array([ri0_b, ri1_b ]) 

            ct = BetaInverse/ri
            s2 = (1.-ct)*(1.+ct) 

            if s2[0] <= 0. and s2[1] <= 0.:
                en = None
            elif s2[0] < 0. and s2[1] > 0.:
                en[0] = (s2[1]*en[0] - s2[0]*en[1])/(s2[1] - s2[0])
                s2[0] = 0.
            elif s2[0] >= 0. and s2[1] >= 0.:
                pass
            elif s2[0] > 0. and s2[1] < 0.:
                en[1] = (s2[1]*en[0] - s2[0]*en[1])/(s2[1] - s2[0]) 
                s2[1] = 0. 
            else:
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
                s2integral +=  (en[1] - en[0])*(s2[0] + s2[1])*0.5
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
        xrg = self.xrg
        cs2e = self.cs2e
        ed = self.ed
        look_ = lambda bi,u:np.interp(u, cs2e[bi], ed[bi] )

        l = {}
        for bi in bis:
            if xrg[bi] is None: continue
            u = np.random.rand(1000000)   
            l[bi] = look_(bi,u) 
        pass
        self.l = l 

    def save_lookup_samples(self, bis):
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
            ax.plot( h[1][:-1], h[0][0:], drawstyle="steps-post", label="bi %6.2f " % bi)  
            ax.legend()
        pass
        fig.show()



if __name__ == '__main__':

    plt.ion()
    ok = opticks_main()

    ckr = CKRindex()
    #bis = np.array(  [1.5,1.6,1.7] )
    #bis = np.array(  [1.6] )
    #bis = np.array(  [1.457] )   ## with one crossing, need to form a range with one side or the other depending on rindex at edges


    bis = np.linspace(1.,ckr.nMin,10)
    #bis = np.linspace(ckr.nMin,ckr.nMax,10)

    ckr.find_energy_range(bis)
    ckr.find_energy_range_plot(bis)

    ckr.s2_cumsum(bis)
    ckr.s2_cumsum_plot(bis)

    ckr.make_lookup_samples(bis)
    #ckr.save_lookup_samples(bis)        # allows chi2 comparison using ana/wavelength_cfplot.py 
    ckr.make_lookup_samples_plot(bis)

    ckr.s2_integrate(bis)
    ckr.s2_integrate_plot(bis)

    
    #ckr.comparison_plot(bis[5:6])
    ckr.comparison_plot(bis[6:7])

