#!/usr/bin/env python
"""
wavelength.py
===============

w0
    qudarap/tests/QCtxTest wavelength samples from GPU texture   

w1
    /tmp/G4OpticksAnaMgr/wavelength.npy from the horses mouth DsG4Scintillation 
    collected by G4OpticksAnaMgr at BeginOfRun

w2
    np.interp results from the icdf obtained from GScintillatorLib, this
    closely matches w0 showing that the GPU texture lookups is working fine.

But w0 and w2 clearly exhibit some artifact of 20nm binning 
in the preparation of the icdf. 

TODO: look into GScintillatorLib is should be using all the information 
provided from material properties, not using coarse bins. 

"""
import os, numpy as np
from opticks.ana.key import keydir
from matplotlib import pyplot as plt 

if __name__ == '__main__':
    fold = "/tmp/QCtxTest"
    w0 = np.load(os.path.join(fold, "wavelength.npy"))

    path1 = "/tmp/G4OpticksAnaMgr/wavelength.npy"
    w1 = np.load(path1) if os.path.exists(path1) else None

    ok = os.environ["OPTICKS_KEY"]
    kd = keydir(ok)
   
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



