#!/usr/bin/env python
"""
::

    ipython -i ck.py 

See ana/steps.py to understand why drawstyle="steps-post" is 
appropriate for rindex related plotting as rindex appears to artificially dupe 
the last value to give equal number of "values" to "edges".

https://arxiv.org/pdf/1206.5530.pdf

Calculation of the Cherenkov light yield from low energetic secondary particles
accompanying high-energy muons in ice and water with Geant4 simulations


"""
import os, logging, numpy as np
import matplotlib.pyplot as plt
from opticks.ana.main import opticks_main
from opticks.ana.key import keydir
from opticks.ana.nload import np_load

log = logging.getLogger(__name__)


class CK(object):
    FIGPATH="/tmp/ck/ck_rejection_sampling.png" 
    PATH="/tmp/ck/ck_%d.npy" 

    kd = keydir()
    rindex_path = os.path.join(kd, "GScintillatorLib/LS_ori/RINDEX.npy")

    #random_path = os.path.expandvars("/tmp/$USER/opticks/TRngBufTest_0.npy")
    random_path="/tmp/QCtxTest/rng_sequence_f_ni1000000_nj16_nk16_tranche100000"

    def init_random(self, num): 

        rnd, paths = np_load(self.random_path)
        if len(paths) == 0:
            log.fatal("failed to find any precooked randoms, create them with : TEST=F QSimTest")
            assert 0 
        pass
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

        self.cursors = cursors
        self.rnd_paths = paths
        self.rnd = rnd 
        self.num = num 


    def init_rindex(self, BetaInverse):

        rindex = np.load(self.rindex_path)
        rindex[:,0] *= 1e6    # into eV 
        rindex_ = lambda ev:np.interp( ev, rindex[:,0], rindex[:,1] ) 

        Pmin = rindex[0,0]  
        Pmax = rindex[-1,0]  
        nMax = rindex[:,1].max() 

        maxCos = BetaInverse / nMax
        maxSin2 = (1.0 - maxCos) * (1.0 + maxCos)

        smry = "nMax %6.4f BetaInverse %6.4f maxCos %6.4f maxSin2 %6.4f" % (nMax, BetaInverse, maxCos, maxSin2) 
        print(smry)

        self.BetaInverse = BetaInverse
        self.maxCos = maxCos
        self.maxCosi = 1. - maxCos
        self.maxSin2 = maxSin2 

        self.rindex = rindex
        self.Pmin = Pmin
        self.Pmax = Pmax
        self.nMax = nMax
        
        self.rindex_ = rindex_
        self.p = np.zeros( (num,4,4), dtype=np.float64 )


    def __init__(self, num=None, BetaInverse=1.5, random=True):
        self.init_rindex(BetaInverse) 
        if random:
            self.init_random(num)
        pass

    def energy_sample_all(self, method="mxs2"):
        for idx in range(self.num):
            self.energy_sample(idx, method=method) 
            if idx % 1000 == 0:
                print(" idx %d num %d " % (idx, self.num))
            pass
        pass

    def stepfraction_sample_all(self):
        for idx in range(self.num):
            self.stepfraction_sample(idx) 
            if idx % 1000 == 0:
                print(" idx %d num %d " % (idx, self.num))
            pass
        pass

    def stepfraction_sample(self, idx):
        """
        What is the expectation for the stepfraction pdf ?
        A linear scaling proportionate to the numbers of
        photons at each end.


            


          G4double NumberOfPhotons, N;

          do { 
             rand = G4UniformRand();
             NumberOfPhotons = MeanNumberOfPhotons1 - rand *
                                    (MeanNumberOfPhotons1-MeanNumberOfPhotons2);
             N = G4UniformRand() *
                            std::max(MeanNumberOfPhotons1,MeanNumberOfPhotons2);
            // Loop checking, 07-Aug-2015, Vladimir Ivanchenko
          } while (N > NumberOfPhotons);



        N = M1 - u (M1 - M2)
          = M1 + u (M2 - M1) 

        """
        MeanNumberOfPhotons = np.array([1000., 1.])
        self.MeanNumberOfPhotons = MeanNumberOfPhotons

        rnd = self.rnd
        cursors = self.cursors  
        uu = rnd[idx].ravel()

        loop = 0 

        NumberOfPhotons = 0.
        N = 0.
        stepfraction = 0.

        while True:
            loop += 1  
            u0 = uu[cursors[idx]]
            cursors[idx] += 1 

            stepfraction = u0 
            NumberOfPhotons = MeanNumberOfPhotons[0] - stepfraction*(MeanNumberOfPhotons[0] - MeanNumberOfPhotons[1])  
            # stepfraction=0  ->  MeanNumberOfPhotons[0]
            # stepfraction=1  ->  MeanNumberOfPhotons[1]

            u1 = uu[cursors[idx]]
            cursors[idx] += 1 

            N = u1*MeanNumberOfPhotons.max()  
            # why is this range not from .min to .max  ?
            # because the sampled number can and will be less that the mean 
            # in some places  

            reject = N > NumberOfPhotons
            if not reject:
                break   
            pass
        pass

        p = self.p[idx]  
        i = self.p[idx].view(np.uint64) 

        p[1,0] = stepfraction
        p[1,1] = NumberOfPhotons
        p[1,2] = N

        p[2,0] = u0
        p[2,1] = u1

        i[3,1] = loop


    def stepfraction_sample_globals(self):
        self.globals( 
            "p",self.p,  
            "stepfraction",    self.p[:,1,0],       
            "NumberOfPhotons", self.p[:,1,1],
            "N",               self.p[:,1,2],

            "u0",              self.p[:,2,0],
            "u1",              self.p[:,2,1],
            "loop",            self.p.view(np.uint64)[:,3,1]
        )

    def stepfraction_plot(self):

        self.stepfraction_sample_globals()

        fdom = np.linspace(0,1,100)
        frg = np.array( [0, 1])
        nrg = self.MeanNumberOfPhotons    
        xf_ = lambda f:np.interp(f, frg, nrg ) 
        self.xf_ = xf_ 

        h_stepfraction = np.histogram( stepfraction )
        h_loop = np.histogram(loop, np.arange(loop.max()+1))  

        title = "ana/ck.py:stepfraction_plot : sampling stepfraction between extremes MeanNumberOfPhotons : %s " % repr(self.MeanNumberOfPhotons)

        fig, axs = plt.subplots(2,3, figsize=ok.figsize )
        plt.suptitle(title)

        ax = axs[0,0]
        ax.scatter( u0, u1, label="(u0,u1)", s=0.1 )
        ax.legend()

        ax = axs[0,1]
        ax.scatter( NumberOfPhotons, N, label="(NumberOfPhotons,N)", s=0.1 )
        ax.legend()

        ax = axs[0,2]
        h = h_stepfraction
        ax.plot( h[1][:-1], h[0], label="h_stepfraction", drawstyle="steps-post" )

        scale = h[0][0]/xf_(0)
        ax.plot( fdom, scale*xf_(fdom), label="xf_ scaled to hist"  )
        ax.legend()

        ax = axs[1,0]
        h = h_loop
        ax.plot( h[1][:-1], h[0], label="h_loop", drawstyle="steps-post" )
        ax.legend()


        ax = axs[1,1]
        ax.plot( fdom, xf_(fdom), label="xf_"  )
        ax.legend()

        fig.show()



    def energy_sample(self, idx, method="mxs2"):
        """
        Why the small difference between s2 when sampling and "expectation-interpolating" 
        in energy regions far from achoring points ? 

        The difference is also visible in ct but less clearly.
        Comparising directly the sampled rs and rindex its difficult 
        to see any difference. 

        When sampling the energy is a random value taked from a flat 
        energy distribution and interpolated individually to give the 
        refractive index.

        When "expectation-interpolating" the energy domain is an abstract analytic ideal
        sort of like a "sample" taken from an infinity of possible values.

        """
        rnd = self.rnd
        rindex = self.rindex
        rindex_ = self.rindex_
        cursors = self.cursors  
        num = self.num
        Pmin = self.Pmin
        Pmax = self.Pmax  
        nMax = self.nMax
        BetaInverse = self.BetaInverse
        maxSin2 = self.maxSin2
        maxCosi = self.maxCosi

        uu = rnd[idx].ravel()

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

            if method == "mxs2": 
                u1_maxSin2 = u1*maxSin2
                keep_sampling = u1_maxSin2 > sin2Theta
            elif method == "mxct":  ## CANNOT DO THIS : MUST USE THE "CONTROLLING" S2 PDF
                u1_maxCosi = u1*maxCosi
                keep_sampling = u1_maxCosi > 1.-cosTheta
            else:
                assert 0
            pass

            loop += 1  

            if dump:
                fmt = "method %s idx %5d u0 %10.5f sampledEnergy %10.5f sampledRI %10.5f cosTheta %10.5f sin2Theta %10.5f u1 %10.5f"
                vals = (method, idx, u0, sampledEnergy, sampledRI, cosTheta, sin2Theta, u1 )
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

    def globals(self, *args):
        assert len(args) % 2 == 0
        for i in range(len(args)//2):
            k = args[2*i+0]
            v = args[2*i+1]
            print("%10s : %s " % (k, str(v.shape)))
            globals()[k] = v 
        pass 

    def energy_sample_globals(self):
        p = self.p
        u0 = p[:,2,0]   
        u1 = p[:,2,1] 
       
        en = p[:,0,0]
        wl = p[:,0,1]
        rs = p[:,0,2]
        ct = p[:,0,3]

        s2 = p[:,1,0]

        self.globals(
           "p",p,
           "u0",u0,
           "u1",u1,
           "en",en,
           "ct",ct,
           "s2",s2,
           "rs",rs
         )

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
    plt.ion()

    ok = opticks_main()
    kd = keydir(os.environ["OPTICKS_KEY"])
 
    num = 10000   # 10k    s2 deviations visible with these stats 
    #num = 100000  # 100k   still visible
    #num = 1000000 # 1M      still visible
    #num = None
    ck = CK(num, BetaInverse=1.5, random=False)

    ck.test_GetAverageNumberOfPhotons(1.78)
    #ck.scan_GetAverageNumberOfPhotons()

    #ck.stepfraction_sample_all()
    #ck.stepfraction_plot()


    enplot = False

if enplot:
    method = "mxs2"
    #method = "mxct"
    ck.energy_sample_all(method=method)
    ck.save()
    ck.energy_sample_globals()

    ri  = ck.rindex 
    edom = ri[:,0] 

    #en_lim = np.array([edom[0],edom[-1]])
    en_lim = np.array([2,10])    ## hmm need to find s2_ roots 

    s2_lim = np.array([-0.01, 0.31])
    ct_lim = np.array([ 0.83, 1.01])
    rs_lim = np.array([ ri[:,1].min(), ri[:,1].max() ])

    
    # lambda functions of energy using np.interpolate inside ck.rindex_
    ri_ = ck.rindex_
    ct_ = lambda e:ck.BetaInverse/ri_(e)
    s2_ = lambda e:(1.-ck.BetaInverse/ri_(e))*(1.+ck.BetaInverse/ri_(e)) 

    ri_interp = ri_(edom)   
    ct_interp = ct_(edom)
    s2_interp = (1. - ct_interp)*(1. + ct_interp )

    en_u0 = ck.Pmin+u0*(ck.Pmax-ck.Pmin)
    s2_u1 = ck.maxSin2*u1


    # pick energy  bin look at the s2 sampled within
    # compare with expectations from interpolation 
    # is the deviation a statistical thing : the sampling
    # can only ever approach the expectation never getting there without 
    # infinite statistics  
    ebin = [5,5.1]
    a_s2 = s2[np.logical_and(en > ebin[0], en < ebin[1])]  
    a_s2_0 = s2_(ebin[0])
    a_s2_1 = s2_(ebin[1])


if enplot:
    fig, ax = plt.subplots(figsize=ok.figsize) 
    fig.suptitle("s2 vs en : deviation between sampling and interpolated, more further from anchor points")

    ax.scatter( en, s2, s=0.1, label="sampled en vs s2") 

    ax.set_xlabel("en")
    ax.set_ylabel("s2")
    ax.set_xlim( en_lim )
    ax.set_ylim( s2_lim )
    xlim = ax.get_xlim()

    ax.plot( ri[:,0], s2_interp, label="s2_interp", color="r" )
    ax.scatter( ri[:,0] , s2_(ri[:,0]), color="b", label="en s2" ) 


    ax.set_xlim(xlim) 
    ylim = ax.get_ylim()
    #for e in ri[:,0]:
    #    ax.plot( [e,e], ylim , linestyle="dotted", color="b") 
    #pass 
    for i in list(range(len(ri)-1)):
        e0,v0 = ri[i]
        e1,v1 = ri[i+1]
        ax.plot( [e0,e1], [s2_(e0), s2_(e1)], linestyle="dotted", color="b" )
        #ax.plot( [e,e], ylim , linestyle="dotted", color="b") 
    pass 

    ax.plot( xlim, [0.,0.], linestyle="dotted", color="r" )
    ax.legend()

    fig.show()



if enplot:
    fig, ax = plt.subplots(figsize=ok.figsize) 
    fig.suptitle("1-ct vs en : deviation between sampling and interpolated, more further from anchor points")

    ax.scatter( en, 1-ct, s=0.1, label="sampled en vs 1-ct") 

    ax.set_xlabel("en")
    ax.set_ylabel("1-ct")
    ax.set_xlim( en_lim )
    ax.set_ylim( 1-ct_lim[::-1] )
    xlim = ax.get_xlim()

    ax.plot( ri[:,0], 1 - ct_interp, label="1-ct_interp", color="r" )
    ax.scatter( ri[:,0] , 1 - ct_(ri[:,0]), color="b", label="en 1-ct" ) 


    ax.set_xlim(xlim) 
    ylim = ax.get_ylim()
    #for e in ri[:,0]:
    #    ax.plot( [e,e], ylim , linestyle="dotted", color="b") 
    #pass 
    for i in list(range(len(ri)-1)):
        e0,v0 = ri[i]
        e1,v1 = ri[i+1]
        ax.plot( [e0,e1], [1-ct_(e0), 1-ct_(e1)], linestyle="dotted", color="b" )
        #ax.plot( [e,e], ylim , linestyle="dotted", color="b") 
    pass 

    ax.plot( xlim, [0.,0.], linestyle="dotted", color="r" )
    ax.legend()

    fig.show()




if enplot:
    fig, ax = plt.subplots(figsize=ok.figsize) 
    fig.suptitle("rs/ri vs en : deviation between sampling and interpolated, more further from anchor points")

    ax.scatter( en, rs, s=0.1, label="sampled en vs rs") 

    ax.set_xlabel("en")
    ax.set_ylabel("rs")
    ax.set_xlim( en_lim )
    ax.set_ylim( rs_lim )
    xlim = ax.get_xlim()

    ax.plot(    ri[:,0],  ri_(ri[:,0]), label="ri", color="r" )
    ax.scatter( ri[:,0] , ri[:,1],      label="en ri", color="b" ) 


    ax.set_xlim(xlim) 
    ylim = ax.get_ylim()
    #for e in ri[:,0]:
    #    ax.plot( [e,e], ylim , linestyle="dotted", color="b") 
    #pass 
    for i in list(range(len(ri)-1)):
        e0,v0 = ri[i]
        e1,v1 = ri[i+1]
        ax.plot( [e0,e1], [ri_(e0), ri_(e1)], linestyle="dotted", color="b" )
        #ax.plot( [e,e], ylim , linestyle="dotted", color="b") 
    pass 

    ax.plot( xlim, [0.,0.], linestyle="dotted", color="r" )
    ax.legend()
    fig.show()









if 0:

    fig, axs = plt.subplots(2,3,figsize=ok.figsize) 

    title = "\n".join(
          ["Cerenkov Rejection Sampling, for JUNO LS refractive index (en:energy in eV ct:cosTheta s2:sin2Theta) ", 
           "2d plots:  (u0,u1) (en,ct) (en,ct)        (Pmin+u0*(Pmax-Pmin),u1*maxSin2) (s2,en) (s2,en) ",
           "RHS compares with interpolated expectation"
          ])  

    fig.suptitle(title) 

    ax = axs[0,0]
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel("u0")
    ax.set_ylabel("u1")
    ax.scatter( u0, u1, s=0.1) 
    ax.set_aspect('equal')  

    ax = axs[1,0]
    ax.set_xlabel("en_u0 : Pmin+u0*(Pmax-Pmin)")
    ax.set_ylabel("s2_u1 : u1*maxSin2")
    ax.set_ylim( s2_lim )
    ax.scatter( en_u0, s2_u1, s=0.1) 
    xlim = ax.get_xlim()
    ax.plot( xlim, [ck.maxSin2, ck.maxSin2] , linestyle="dotted", color="r") 

    ax = axs[0,1]
    ax.scatter( en, 1.-ct, s=0.1) 
    ax.set_xlabel("en")
    ax.set_ylabel("1-ct")
    ax.set_ylim( 1-ct_lim[::-1] ) 
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for e in ck.rindex[:,0]:
        ax.plot( [e,e], ylim , linestyle="dotted", color="b") 
    pass
    ax.set_xlim(xlim)
    ax.plot( xlim, [1.-ck.maxCos, 1.-ck.maxCos] , linestyle="dotted", color="r") 
   
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
    #ax.plot( ck.rindex[:,0], ck.BetaInverse/ck.rindex[:,1], drawstyle="steps-post" )
    ax.plot( ck.rindex[:,0], ct_interp, color="r" )  
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
    ax.plot( ck.rindex[:,0], s2_interp, label="s2_interp", color="r" )
    ax.set_xlim(xlim) 
    ylim = ax.get_ylim()
    for e in ck.rindex[:,0]:
        ax.plot( [e,e], ylim , linestyle="dotted", color="b") 
    pass 
    ax.plot( xlim, [0.,0.], linestyle="dotted", color="r" )


    fig.show() 
    path = ck.FIGPATH
    print("save to %s " % path)
    fig.savefig(path)






if 0:
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

    
