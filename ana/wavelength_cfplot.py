#!/usr/bin/env python
"""
wavelength_cfplot.py
=======================

::

    an ; 

    ARG=0 ipython -i wavelength_cfplot.py
    ARG=1 ipython -i wavelength_cfplot.py
    ARG=2 ipython -i wavelength_cfplot.py
    ARG=3 ipython -i wavelength_cfplot.py

    ARG=4 ipython -i wavelength_cfplot.py
    ARG=5 ipython -i wavelength_cfplot.py
    ARG=6 ipython -i wavelength_cfplot.py


    mkdir -p ~/simoncblyth.bitbucket.io/env/presentation/ana/wavelength_cfplot
    cp /tmp/ana/wavelength_cfplot/*.png ~/simoncblyth.bitbucket.io/env/presentation/ana/wavelength_cfplot/


"""

import os, numpy as np, logging
log = logging.getLogger(__name__)

from opticks.ana.main import opticks_main 
from opticks.ana.key import keydir
from opticks.ana.material import Material
from opticks.ana.wavelength import Wavelength
from opticks.ana.cfplot import one_cfplot
from opticks.ana.cfh import CFH

from matplotlib import pyplot as plt 

if __name__ == '__main__':
    ok = opticks_main()
    kd = keydir(os.environ["OPTICKS_KEY"])
    wl = Wavelength(kd)

    arg = int(os.environ.get("ARG","0")) 
  
    if arg == 0:
        a, b = wl.get_keys('DsG4Scintillator_G4OpticksAnaMgr', "Opticks_QCtxTest_hd20") 
    elif arg == 1:
        a, b = wl.get_keys('DsG4Scintillator_G4OpticksAnaMgr', "Opticks_QCtxTest_hd0") 
    elif arg == 2:
        a, b = wl.get_keys('DsG4Scintillator_G4OpticksAnaMgr', 'Opticks_QCtxTest_hd20_cudaFilterModePoint') 
    elif arg == 3:
        a, b = wl.get_keys('DsG4Scintillator_G4OpticksAnaMgr', 'Opticks_QCtxTest_hd0_cudaFilterModePoint')
    elif arg == 4:
        a, b = wl.get_keys('G4Cerenkov_modified_SKIP_CONTINUE', 'ck_photon' )
    elif arg == 5:
        a, b = wl.get_keys('G4Cerenkov_modified_SKIP_CONTINUE_10k', 'ck_photon_10k' )
    elif arg == 6:
        a, b = wl.get_keys('G4Cerenkov_modified_SKIP_CONTINUE_3M', 'ck_photon' )
    else:
        assert 0
    pass

    h = CFH()

    if arg < 4:
        h.log = True 
        h.ylim = (50., 5e4 )
        title = "Compare two 1M sample LS scintillation wavelength distributions in 1nm bins" 
        xline = [wl.interp(0.05), wl.interp(0.95)]
    else:
        h.log = True
        title = "Compare two 2.82M samples of Cerenkov wavelength distributions in 1nm bins : poor chi2, interpolation artifact ?  " 
        xline = []
    pass 

    h.suptitle = title
    c2cut = 10 
    h(wl.dom[:-1], wl.w[a], wl.w[b], [wl.l[a], wl.l[b]], c2cut=c2cut )

    fig, axs = one_cfplot(ok, h, xline=xline )

    if arg in [4,5]:

        rindex = np.load(os.path.join(kd, "GScintillatorLib/LS_ori/RINDEX.npy"))
        rindex[:,0] *= 1e6   
        rindex[:,0] = 1240./rindex[:,0]
        rindex = rindex[::-1]       

        ax = axs[0]

        ylim = ax.get_ylim()
        for i in range(len(rindex)):
            ax.plot( [rindex[i,0], rindex[i,0]], ylim , linestyle="dotted", color="b" )
        pass

        axr = ax.twinx() 
        axr.set_ylabel("rindex")
        axr.spines['right'].set_position(('outward', 0))

        p3, = axr.plot( rindex[:,0] , rindex[:,1], drawstyle="steps", label="rindex", color="b" )

        #wmin, wmax = 100., 400.
        wmin, wmax = 80., 800.
        
        axs[0].set_xlim(wmin,wmax)
        axs[1].set_xlim(wmin,wmax)

        axs[1].set_ylim(0,1)
    pass

    plt.ion()
    plt.show()

    print(h.lhabc)   

    path="/tmp/ana/wavelength_cfplot/%s_%s.png" % ( wl.l[a], wl.l[b] )
    fold=os.path.dirname(path)
    if not os.path.isdir(fold):
       os.makedirs(fold)
    pass 
    log.info("save to %s " % path)
    fig.savefig(path)


