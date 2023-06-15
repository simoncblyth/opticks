#!/usr/bin/env python

import os, logging, numpy as np
from opticks.sysrap.sevt import SEvt

log = logging.getLogger(__name__)
GLOBAL = int(os.environ.get("GLOBAL","0")) == 1
MODE = int(os.environ.get("MODE","3")) 

if MODE in [2,3]:
    from opticks.ana.pvplt import *   
    # HMM this import overrides MODE, so need to keep defaults the same 
pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("GLOBAL:%d MODE:%d" % (GLOBAL,MODE))

    e = SEvt.Load(symbol="e")
    print(repr(e))

    qtab = e.minimal_qtab()


    label = e.f.base

    if MODE in [0,1]:
        print("not plotting as MODE %d in environ" % MODE )
    elif MODE == 2:
        pl = mpplt_plotter(label=label)
        fig, axs = pl
        assert len(axs) == 1
        ax = axs[0]
    elif MODE == 3:
        pl = pvplt_plotter(label)
        pvplt_viewpoint(pl)   # sensitive to EYE, LOOK, UP envvars
        pvplt_frame(pl, e.f.sframe, local=not GLOBAL )
    pass

    #pp = e.f.inphoton[:,0,:3]
    #pp = e.f.photon[:,0,:3]
    pp = e.f.record[:,:,0,:3].reshape(-1,3)

    gpos = np.ones( [len(pp), 4 ] )
    gpos[:,:3] = pp
    lpos = np.dot( gpos, e.f.sframe.w2m )
    upos = gpos if GLOBAL else lpos

    H,V = 0,2 
  
    if MODE == 2:
        ax.scatter( upos[:,H], pp[:,V], s=0.1 )
    elif MODE == 3:
        pl.add_points(upos[:,:3])
    else:
        pass
    pass

    if MODE == 2:
        fig.show()
    elif MODE == 3:
        pl.show()
    pass
pass


 

