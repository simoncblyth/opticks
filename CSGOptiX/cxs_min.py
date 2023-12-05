#!/usr/bin/env python

import os, logging, numpy as np
from opticks.sysrap.sevt import SEvt, SAB
log = logging.getLogger(__name__)


PLOT = os.environ.get("PLOT","scatter")
GLOBAL = int(os.environ.get("GLOBAL","0")) == 1
MODE = int(os.environ.get("MODE","3")) 
SEL = int(os.environ.get("SEL","0")) 


if MODE in [2,3]:
    from opticks.ana.pvplt import *   
    # HMM this import overrides MODE, so need to keep defaults the same 
pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("GLOBAL:%d MODE:%d SEL:%d" % (GLOBAL,MODE, SEL))

    a = SEvt.Load("$AFOLD", symbol="a")
    print(repr(a))

    if "BFOLD" in os.environ:   
        b = SEvt.Load("$BFOLD", symbol="b") 
        print(repr(b))
        ab = SAB(a,b) 
        print(repr(ab))
    pass


    e = a 

    qtab = e.minimal_qtab()
    print("qtab")
    print(qtab)


    if hasattr(e.f, 'record'):
        pp = e.f.record[:,:,0,:3].reshape(-1,3)
        found = "record"
    elif hasattr(e.f, 'hit'):
        pp = e.f.hit[:,0,:3]
        found = "hit"
    elif hasattr(e.f, 'photon'):
        pp = e.f.photon[:,0,:3]
        found = "photon"
    elif hasattr(e.f, 'inphoton'):
        pp = e.f.inphoton[:,0,:3]
        found = "inphoton"
    elif hasattr(e.f, 'genstep'):
        pp = e.f.genstep[:,1,:3]
        found = "genstep"
    else:
        pp = None
        found = "NONE" 
        pass
    pass

    assert not pp is None

    gpos = np.ones( [len(pp), 4 ] )
    gpos[:,:3] = pp
    lpos = np.dot( gpos, e.f.sframe.w2m )
    upos = gpos if GLOBAL else lpos

    if MODE in [0,1]:
        print("not plotting as MODE %d in environ" % MODE )
    elif MODE == 2:
        label = "%s : " % ( e.f.base.replace("/data/blyth/opticks/","") )
        if PLOT.startswith("seqnib") and hasattr(e.f, "seqnib"):
            label +=  " : %s " % PLOT
        pass 
        expl = "Photon History Step Counts Occurrence in single 1M photon event"
        title = "\n".join([label, expl])
        pl = mpplt_plotter(label=title)
        fig, axs = pl
        assert len(axs) == 1
        ax = axs[0]
    elif MODE == 3:

        label = "%s : found %s " % ( e.f.base, found )
        print(label)
        pl = pvplt_plotter(label)
        pvplt_viewpoint(pl)   # sensitive to EYE, LOOK, UP envvars
        pvplt_frame(pl, e.f.sframe, local=not GLOBAL )
    pass
    

    H,V = 0,2  # X, Z

    if SEL == 1:
        sel = np.logical_and( np.abs(upos[:,H]) < 500, np.abs(upos[:,V]) < 500 )
        spos = upos[sel]
    else:
        spos = upos 
    pass

    if MODE == 2:

        if PLOT.startswith("scatter"):
            ax.scatter( spos[:,H], spos[:,V], s=0.1 )
        elif PLOT.startswith("seqnib") and hasattr(e.f, "seqnib"):
            seqnib = e.f.seqnib  

            bounce = np.arange(len(seqnib))
            ax.set_aspect('auto')
            ax.plot( bounce, seqnib )
            ax.scatter( bounce, seqnib )
            ax.set_ylabel("Photon counts for step points (0->31)", fontsize=20 )

            cs_seqnib = np.cumsum(seqnib)   
            ax2 = ax.twinx() 
            ax2.plot( bounce, cs_seqnib, linestyle="dashed" )
            ax2.set_ylabel( "Cumulative counts rising to total of 1M photons", fontsize=20 )
        elif PLOT.startswith("thit") and hasattr(e.f, "hit"):
            thit = e.f.hit[:,0,3] 
            thit_n, thit_bin = np.histogram(thit, bins=100) 
            ax.set_aspect('auto')
            ax.plot( thit_bin[:-1], thit_n, drawstyle="steps-mid", label="Hit time [ns]" )
            ax.set_yscale('log')
            ax.legend() 
        else:
            print("PLOT:%s unhandled" % PLOT)
        pass 
    elif MODE == 3:
        pl.add_points(spos[:,:3])
    else:
        pass
    pass

    if MODE == 2:
        fig.show()
    elif MODE == 3:
        pl.show()
    pass
pass


 

