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


if __name__ == '__main__':

    ok = opticks_main()
    kd = keydir(os.environ["OPTICKS_KEY"])
 
    ri = np.load(os.path.join(kd, "GScintillatorLib/LS_ori/RINDEX.npy"))
    ri[:,0] *= 1e6  
    ri_ = lambda e:np.interp(e, ri[:,0], ri[:,1] )
    nMax = ri[:,1].max() 
    nMin = ri[:,1].min() 


    #BetaInverse = [1.5,1.6,1.7]
    #BetaInverse = [1.6]
    #BetaInverse = [1.5]
    #BetaInverse = np.linspace(nMin, nMax, 10) 

    #BetaInverse = [1.457]   
    # with one crossing, need to form a range with one side or the other 
    # depending on "rindex - BetaInverse" at domain edges  

    #BetaInverse = np.linspace( 1., nMax, 10)  
    BetaInverse = [1.] 
    # what about zero crossing 

    xri = {}
    xrg = {}
    for bi in BetaInverse:
        xri[bi] = find_cross(ri, BetaInverse=bi)

        lhs = ri[0,1] - bi > 0   # ck allowed at left edge ?
        rhs = ri[-1,1] - bi > 0  # ck allowed at right edge ?  

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

    ## prevent the cos(th) from exceeding 1 in disallowed regions
    ct_ = lambda bi,e:np.minimum(1.,bi/np.interp(e, ri[:,0], ri[:,1] ))
    s2_ = lambda bi,e:(1-ct_(bi,e))*(1+ct_(bi,e))

    plt.ion()
    fig, ax = plt.subplots(figsize=ok.figsize); 

    # steps make no sense for rindex, as it is inherently interpolated between measured points
    ax.plot( ri[:,0], ri[:,1] )  

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.plot( xlim, [nMax,nMax], linestyle="dotted", color="r" )

    for bi in BetaInverse:
        ax.plot( xlim, [bi,bi], linestyle="dotted", color="r" )
    pass  
    ax.plot( [ri[0,0], ri[0,0]], ylim, linestyle="dotted", color="r" )
    ax.plot( [ri[-1,0], ri[-1,0]], ylim, linestyle="dotted", color="r" )

    for bi in BetaInverse:
        for x in xri[bi]:
            ax.plot( [x,x], ylim, linestyle="dotted", color="r" )
        pass
    pass
    ax.scatter( ri[:,0], ri[:,1] )
    fig.show()


if 1:
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

    ed = {}
    s2e = {}
    cs2e = {}

    for bi in BetaInverse:
        if xrg[bi] is None: continue
        ed[bi] = np.linspace(xrg[bi][0],xrg[bi][1],4096)    # energy range from min to max allowable
        s2e[bi] = s2_(bi,ed[bi])  
        cs2e[bi] = np.cumsum(s2e[bi])  
        cs2e[bi] /= cs2e[bi][-1]      # last bin will inevitably be maximum one as cumulative   
    pass 


    fig, axs = plt.subplots(figsize=ok.figsize) 
    for i, bi in enumerate(BetaInverse):
        if xrg[bi] is None: continue
        ax = axs if axs.__class__.__name__ == 'AxesSubplot' else axs[i] 
        ax.plot( ed[bi], cs2e[bi], label="cs2e : integrated s2 vs e bi:%6.3f  " % bi )
        #ax.set_xlim( xrg[1.5][0], xrg[1.5][1] )
        ax.legend()
    pass
    fig.show()


    look_ = lambda bi,u:np.interp(u, cs2e[bi], ed[bi] )

    l = {}
    for bi in BetaInverse:
        if xrg[bi] is None: continue
        u = np.random.rand(1000000)   
        l[bi] = look_(bi,u) 
    pass


    fold = "/tmp/rindex" 
    for i,bi in enumerate(BetaInverse):
        if xrg[bi] is None: continue
        path = os.path.join(fold,"en_integrated_lookup_1M_%d.npy" % i ) 
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        pass
        print("save to %s " % path)
        #np.save(path, l[bi] ) 
    pass

    fig, axs = plt.subplots(figsize=ok.figsize)
    for i, bi in enumerate(BetaInverse):
        if xrg[bi] is None: continue
        ax = axs if axs.__class__.__name__ == 'AxesSubplot' else axs[i] 
        hd = np.arange(xrg[bi][0],xrg[bi][1],0.1)   
        h = np.histogram(l[bi], hd )
        ax.plot( h[1][:-1], h[0][0:], drawstyle="steps-post", label="bi %6.2f " % bi)  
        ax.legend()
    pass
    fig.show()

