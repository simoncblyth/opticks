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
import os, numpy as np, logging
log = logging.getLogger(__name__)

from opticks.ana.main import opticks_main 
from opticks.ana.key import keydir
from opticks.ana.material import Material
from opticks.ana.wavelength import Wavelength

from matplotlib import pyplot as plt 

if __name__ == '__main__':
    ok = opticks_main()
    kd = keydir(os.environ["OPTICKS_KEY"])
    wl = Wavelength(kd)

    h = wl.h
    l = wl.l
    dom = wl.dom[:-1]


    plt.ion()
    fig, ax = plt.subplots(figsize=ok.figsize)

    fig.suptitle("LS : Absorption Length/Scattering Length/Reemission Probability/Spectrum Distribution")

    ax.set_xlabel("wavelength (nm) [2 nm bins]" )
  
    ax.set_ylabel("1M wavelength sample, 2nm bin counts")
    ax.plot( dom, h[0], drawstyle="steps", label=l[0] )  
    ax.plot( dom, h[1], drawstyle="steps", label=l[1] )  
    #ax.plot( dom, h[2], drawstyle="steps", label=l[2] )  
    ax.legend(loc="center left")


    # https://stackoverflow.com/questions/9103166/multiple-axis-in-matplotlib-with-different-scales
    axr = ax.twinx() 
    axr2 = ax.twinx() 

    ls = Material("LS") 
    reempr = ls.reemission_prob(dom) 
    abslen = ls.absorption_length(dom) 
    scatlen = ls.scattering_length(dom) 

    axr.set_ylabel("length (m)")
    p1, = axr.plot( dom, abslen/1000., drawstyle="steps", label="abslen (m)", color="r" )
    p2, = axr.plot( dom, scatlen/1000., drawstyle="steps", label="scatlen (m)", color="g" )

    axr2.set_ylabel("Re-emission Probability")
    axr2.spines['right'].set_position(('outward', 60))
    p3, = axr2.plot( dom, reempr, drawstyle="steps", label="reemprob", color="b" )

    axr.legend(handles=[p1,p2,p3], loc="best")

    lines = False
    if lines:
        ylim = np.array(ax.get_ylim())
        for w in [410,430]:
            ax.plot( [w,w], ylim*1.1 )    
        pass
    pass

    ax.axvspan(410, 430, color='grey', alpha=0.3, lw=0)
 
    ax.text( 420, 100, "410-430 nm", ha="center" )

    fig.tight_layout()
    plt.show()

    path="/tmp/ana/LS_wavelength.png"
    fold=os.path.dirname(path)
    if not os.path.isdir(fold):
       os.makedirs(fold)
    pass 
    log.info("save to %s " % path)
    fig.savefig(path)



