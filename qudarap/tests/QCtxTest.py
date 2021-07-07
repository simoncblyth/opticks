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

class QCtxTest(object):
    FOLD = "/tmp/QCtxTest"
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

    def cerenkov_wavelength(self):
        w0 = np.load(os.path.join(self.FOLD, "wavelength_cerenkov.npy"))

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
        self.p = p
        self.w = w
 

if __name__ == '__main__':
    qc = QCtxTest()    
    #qc.scint_wavelength()
    #qc.boundary_lookup_all() 
    qc.boundary_lookup_line() 
    p = qc.p
    w = qc.w
 
    fig, ax = plt.subplots()

    #ax.plot( w, p[:,0], drawstyle="steps", label="ri" )
    #ax.plot( w, p[:,1], drawstyle="steps", label="abslen" )
    #ax.plot( w, p[:,2], drawstyle="steps", label="scatlen" )
    ax.plot( w, p[:,3], drawstyle="steps", label="reemprob" )

    ax.legend()
    fig.show()






