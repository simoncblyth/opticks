#!/usr/bin/env python
"""
QCtxTest.py
=============

::

    qudarap
    ipython -i tests/QCtxTest.py 



"""
import os, numpy as np
from opticks.ana.key import keydir
from matplotlib import pyplot as plt 


class PropLookup(object):
    FOLD = "/tmp/QCtxTest"
    def __init__(self):
        pp = np.load(os.path.join(self.FOLD,"prop_lookup_pp.npy"))
        x = np.load(os.path.join(self.FOLD,"prop_lookup_x.npy"))
        yy = np.load(os.path.join(self.FOLD,"prop_lookup_yy.npy"))

        self.pp = pp 
        self.x = x 
        self.yy = yy 

        colors = "rgbcmyk"
        
        fig, ax = plt.subplots(figsize=[12.8,7.2])
        for i,y in enumerate(yy):
            ax.plot( x, y, color=colors[i] ) 
        pass
        for i,p in enumerate(pp):
            pr = p.reshape(-1,2)
            ax.scatter( pr[:,0], pr[:,1], color=colors[i] ) 
        pass
        fig.show()




class QCtxTest(object):
    FOLD = "/tmp/QCtxTest"
    hc_eVnm = 1240. 

    @classmethod
    def LoadCK(cls, num=10000):
        path = os.path.join( cls.FOLD, "cerenkov_photon_%d.npy" % num )
        p = np.load(path)
        return p 

    def scint_wavelength(self):
        """
        See::
 
             ana/wavelength.py
             ana/wavelength_cfplot.py

        """
        w0 = np.load(os.path.join(self.FOLD, "wavelength_scint_hd20.npy"))

        path1 = "/tmp/G4OpticksAnaMgr/wavelength.npy"
        w1 = np.load(path1) if os.path.exists(path1) else None

        kd = keydir(os.environ["OPTICKS_KEY"])
        aa = np.load(os.path.join(kd,"GScintillatorLib/GScintillatorLib.npy"))
        a = aa[0,:,0]
        b = np.linspace(0,1,len(a))
        u = np.random.rand(1000000)  
        w2 = np.interp(u, b, a )  

        #bins = np.arange(80, 800, 4)  
        bins = np.arange(300, 600, 4)  

        h0 = np.histogram( w0 , bins )
        h1 = np.histogram( w1 , bins )
        h2 = np.histogram( w2 , bins )

        fig, ax = plt.subplots()
     
        ax.plot( bins[:-1], h0[0], drawstyle="steps-post", label="OK.QCtxTest" )  
        ax.plot( bins[:-1], h1[0], drawstyle="steps-post", label="G4" )  
        ax.plot( bins[:-1], h2[0], drawstyle="steps-post", label="OK.GScint.interp" )  

        ylim = ax.get_ylim()

        for w in [320,340,360,380,400,420,440,460,480,500,520,540]:
            ax.plot( [w,w], ylim )    
        pass

        ax.legend()

        plt.show()

        self.w0 = w0
        self.w1 = w1
        self.w2 = w2

    def boundary_lookup_all(self):
        l = np.load(os.path.join(self.FOLD, "boundary_lookup_all.npy"))
        s_ = np.load(os.path.join(self.FOLD, "boundary_lookup_all_src.npy"))
        s = s_.reshape(l.shape)  
        assert np.allclose(s, l)   

        self.s_ = s_ 
        self.s = s 
        self.l = l 

    def boundary_lookup_line(self):
        p = np.load(os.path.join(self.FOLD, "boundary_lookup_line_props.npy"))
        w = np.load(os.path.join(self.FOLD, "boundary_lookup_line_wavelength.npy"))

        path = "/tmp/RINDEXTest/g4_line_lookup.npy"
        g = np.load(path) if os.path.exists(path) else None 

        fig, ax = plt.subplots()
        ax.plot( w, p[:,0], drawstyle="steps", label="ri.qctx" )

        if not g is None:
            assert np.all( w  == g[:,0] ) 
            ax.plot( w, g[:,1], drawstyle="steps", label="ri.g4" ) 
            ri_diff = p[:,0] - g[:,1] 
            self.ri_diff = ri_diff
            print("ri_diff.min %s ri_diff.max %s " % (ri_diff.min(), ri_diff.max()))
        pass

        #ax.plot( w, p[:,1], drawstyle="steps", label="abslen" )
        #ax.plot( w, p[:,2], drawstyle="steps", label="scatlen" )
        #ax.plot( w, p[:,3], drawstyle="steps", label="reemprob" )

        ax.legend()
        fig.show()

        self.p = p
        self.w = w
        self.g = g 



    def cerenkov_photon(self):
        name = "cerenkov_photon"
        p = np.load(os.path.join(self.FOLD, "%s.npy" % name))
        self.p = p         

        en = p[0,0,0]
        wl = p[0,0,1]
        ri = p[0,0,2]
        ct = p[0,0,3]

        s2 = p[0,1,0]
        bi = p[0,1,3]

        w0 = p[0,2,0]
        w1 = p[0,2,1]
        u0 = p[0,2,2] 
        u1 = p[0,2,3]  

        li = p[0,3,0].view(np.int32)
        lo = p[0,3,1].view(np.int32)



    def rng_sequence(self):
        name = "rng_sequence"
        r = np.load(os.path.join(self.FOLD, "%s.npy" % name))
        self.r = r         

        fig, axs = plt.subplots()
        fig.suptitle(name) 

        r_dom = np.linspace(0,1,100)
        h_r = np.histogram(r, r_dom )

        ax = axs
        ax.plot( h_r[1][:-1], h_r[0], label="h_r", drawstyle="steps" )

        ax.set_ylim( 0, h_r[0].max()*2. )

        ax.legend()

        fig.show()
         



    def cerenkov_wavelength(self):

        nm_dom = [80, 800, 1]

        name = "wavelength_cerenkov"
        w0 = np.load(os.path.join(self.FOLD, "%s.npy" % name))
        e0 = self.hc_eVnm/w0 
        wdom = np.arange(*nm_dom)

        edom0 = self.hc_eVnm/wdom[::-1]      
        # Convert to energy and reverse order for ascending energy domain
        # but thats a funny variable energy bin domain (with fixed 1 nm wavelength bin) 
        #
        # Instead use equal energy bin domain across same range as wavelenth one
        # with same number of bins.
        edom = np.linspace( self.hc_eVnm/nm_dom[1], self.hc_eVnm/nm_dom[0], len(wdom) )  


        h_w0 = np.histogram(w0, wdom )
        h_e0 = np.histogram(e0, edom )

        h_w100 = np.histogram(w0, 100 )
        h_e100 = np.histogram(e0, edom )



        fig, axs = plt.subplots(2,2)
        fig.suptitle(name) 

        ax = axs[0,0]
        ax.plot( h_w0[1][:-1], h_w0[0], label="h_w0", drawstyle="steps" )
        ax.legend()

        ax = axs[0,1]
        ax.plot( h_e0[1][:-1], h_e0[0], label="h_e0", drawstyle="steps" )
        ax.legend()

        ax = axs[1,0]
        ax.plot( h_w100[1][:-1], h_w100[0], label="h_w100", drawstyle="steps" )
        ax.legend()

        ax = axs[1,1]
        ax.plot( h_e100[1][:-1], h_e100[0], label="h_e100", drawstyle="steps" )
        ax.legend()



        fig.show()

        self.w0 = w0
        self.e0 = e0
        self.h_w0 = h_w0 
        self.h_e0 = h_e0 

        self.edom = edom
 

if __name__ == '__main__':
    #q = QCtxTest()    
    #q.scint_wavelength()
    #q.boundary_lookup_all() 
    #q.boundary_lookup_line() 
    #q.cerenkov_wavelength() 
    #q.cerenkov_photon() 
    #q.rng_sequence() 

    t = PropLookup()

 






