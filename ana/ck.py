#!/usr/bin/env python
"""
::

    ipython -i ck.py 

See ana/steps.py to understand why drawstyle="steps-post" is 
appropriate for rindex related plotting as rindex appears to artificially dupe 
the last value to give equal number of "values" to "edges".
"""
import os, logging, numpy as np
import matplotlib.pyplot as plt
from opticks.ana.main import opticks_main
from opticks.ana.key import keydir
from opticks.ana.nload import np_load

log = logging.getLogger(__name__)


class CK(object):
    PATH="/tmp/ck/ck_%d.npy" 

    kd = keydir()
    rindex_path = os.path.join(kd, "GScintillatorLib/LS_ori/RINDEX.npy")

    #random_path = os.path.expandvars("/tmp/$USER/opticks/TRngBufTest_0.npy")
    #random_path="/tmp/QCtxTest/rng_sequence_f_ni1000000_nj16_nk16_tranche100000"
    random_path="/tmp/QCtxTest/rng_sequence_f"

    def __init__(self, num=None):
        rnd, paths = np_load(self.random_path)
        if num is None:
            num = len(rnd)
        else:
            enough = num <= len(rnd)
            if not enough:
                log.fatal("not enough precooked randoms len(rnd) %d num %d " % (len(rnd), num))
            pass
            assert enough
        pass
        cursors = np.zeros( num, dtype=np.int32 ) 

        self.rnd_paths = paths
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
            if idx % 1000 == 0:
                print(" idx %d num %d " % (idx, self.num))
            pass
        pass

    def energy_sample(self, idx, BetaInverse=1.5):

        self.BetaInverse = BetaInverse

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

        self.maxSin2 = maxSin2 

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

        p[2,0] = u0
        p[2,1] = u1

        i[3,1] = loop

    def save(self):
        path = self.PATH % self.num
        fold = os.path.dirname(path)
        if not os.path.exists(fold):
            os.makedirs(fold)
        pass
        log.info("save to %s " % path)
        np.save(path, self.p)

    @classmethod
    def Load(cls, num):
        path = cls.PATH % num
        return np.load(path)

 

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    ok = opticks_main()
    kd = keydir(os.environ["OPTICKS_KEY"])
 
    num = 10000 
    #num = None
    ck = CK(num)

if 1:
    ck.energy_sample_all(BetaInverse=1.5)
    ck.save()

    p = ck.p

if 1:
    u0 = p[:,2,0]   
    u1 = p[:,2,1] 
   
    en = p[:,0,0]
    ct = p[:,0,3]
    s2 = p[:,1,0]

    plt.ion()
    fig, axs = plt.subplots(2,3,figsize=ok.figsize) 

    ax = axs[0,0]
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel("u0")
    ax.set_ylabel("u1")
    ax.scatter( u0, u1, s=0.1) 
    ax.set_aspect('equal')  


    s2_lim = -0.01, 0.31
    ct_lim =  0.83, 1.01 

    ax = axs[1,0]
    ax.set_xlabel("Pmin+u0*(Pmax-Pmin)")
    ax.set_ylabel("u1*maxSin2")
    ax.set_ylim( s2_lim )

    x = ck.Pmin+u0*(ck.Pmax-ck.Pmin)
    y = ck.maxSin2*u1
    ax.scatter( x, y, s=0.1) 

    ax = axs[0,1]
    ax.scatter( en, ct, s=0.1) 
    ax.set_xlabel("en")
    ax.set_ylabel("ct")
    ax.set_ylim( ct_lim ) 

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for e in ck.rindex[:,0]:
        ax.plot( [e,e], ylim , linestyle="dotted", color="b") 
    pass
    ax.set_xlim(xlim)
   
    ax = axs[1,1]
    ax.scatter( en, s2, s=0.1) 
    ax.set_xlabel("en")
    ax.set_ylabel("s2")
    ax.set_ylim( s2_lim )

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for e in ck.rindex[:,0]:
        ax.plot( [e,e], ylim , linestyle="dotted", color="b") 
    pass
    ax.set_xlim(xlim)
 

    ax = axs[0,2]
    ax.scatter( en, ct, s=0.1) 
    ax.set_xlabel("en")
    ax.set_ylabel("ct")
    ax.set_ylim( ct_lim ) 
    xlim = ax.get_xlim()

    ax.plot( ck.rindex[:,0], ck.BetaInverse/ck.rindex[:,1], drawstyle="steps-post" )
    ax.set_xlim(xlim) 
    ax.plot( xlim, [1.,1.], linestyle="dotted", color="r" )

    ylim = ax.get_ylim()
    for e in ck.rindex[:,0]:
        ax.plot( [e,e], ylim , linestyle="dotted", color="b") 
    pass 

    if 0:
        for v in ck.BetaInverse/ck.rindex[:,1]:
            ax.plot( xlim, [v,v] , linestyle="dotted", color="b") 
        pass 
    pass
        


    ax = axs[1,2]
    ax.scatter( en, s2, s=0.1) 
    ax.set_xlabel("en")
    ax.set_ylabel("s2")
    ax.set_ylim( s2_lim )
    xlim = ax.get_xlim()

    ri_s2 = (1.-ck.BetaInverse/ck.rindex[:,1])*(1.+ck.BetaInverse/ck.rindex[:,1]) 
    ax.plot( ck.rindex[:,0], ri_s2, drawstyle="steps-post", label="ri_s2" )
    ax.set_xlim(xlim) 

    ylim = ax.get_ylim()
    for e in ck.rindex[:,0]:
        ax.plot( [e,e], ylim , linestyle="dotted", color="b") 
    pass 
    ax.plot( xlim, [0.,0.], linestyle="dotted", color="r" )


    fig.show() 


if 1:
    fig3, ax = plt.subplots(figsize=ok.figsize) 

    h = np.histogram( en )
    ax.plot( h[1][:-1], h[0], label="hist en" )
    ax.legend()
    fig3.show()



if 0:
    fig2, ax = plt.subplots(figsize=ok.figsize) 
    fig2.suptitle( repr(ck.rindex.T) )

    ax.scatter( en, ct, s=0.5) 
    ax.set_xlabel("en")
    ax.set_ylabel("ct")
    xlim = ax.get_xlim()

    ax.plot( ck.rindex[:,0], ck.BetaInverse/ck.rindex[:,1], drawstyle="steps-post" )
    ax.set_xlim(xlim) 
    ax.plot( xlim, [1.,1.], linestyle="dotted", color="r" )

    ylim = ax.get_ylim()
    for e in ck.rindex[:,0]:
        ax.plot( [e,e], ylim , linestyle="dotted", color="b") 
    pass 
    for v in ck.BetaInverse/ck.rindex[:,1]:
        ax.plot( xlim, [v,v] , linestyle="dotted", color="b") 
    pass 
    
    fig2.show() 


    

