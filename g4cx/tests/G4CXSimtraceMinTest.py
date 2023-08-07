#!/usr/bin/env python
"""
G4CXSimtraceMinTest.py : simtrace plot backdrop with APID, BPID onephotonplot on top 
=======================================================================================

This is now run by G4CXAppTest.sh with eg::

    APID=62 MODE=2 ~/opticks/g4cx/tests/G4CXAppTest.sh tra

This aims to do similar to G4CXSimtraceTest.py but in a more minimal way,
drawing on developments from cx/cxs_min.py 

"""

import os, logging, numpy as np
from opticks.sysrap.sevt import SEvt, SAB
log = logging.getLogger(__name__)

GLOBAL = int(os.environ.get("GLOBAL","0")) == 1
MODE = int(os.environ.get("MODE","3"))
SEL = int(os.environ.get("SEL","0"))


if MODE in [2,3]:
    from opticks.ana.pvplt import *
    # HMM this import overrides MODE, so need to keep defaults the same 
pass


def onephotonplot(pl, e):
    if e is None: return 
    if e.pid < 0: return

    if not hasattr(e,'r'): return
    if e.r is None: return

    r = e.r 
    off = e.off

    rpos = r[:,0,:3] + off 

    if MODE == 2:
        fig, axs = pl  
        assert len(axs) == 1 
        ax = axs[0]
        if True:
            mpplt_add_contiguous_line_segments(ax, rpos, axes=(H,V), label=None )
            #self.mp_plab(ax, f)
        pass
        #if "nrm" in f.opt:
        #    self.mp_a_normal(ax, f)  
        #pass
    elif MODE == 3:
        pass
        pvplt_add_contiguous_line_segments(pl, rpos )
    pass 




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    a = SEvt.Load("$AFOLD", symbol="a")
    b = SEvt.Load("$BFOLD", symbol="b")
    t = SEvt.Load("$TFOLD", symbol="t")

    print(repr(a))
    print(repr(b))
    print(repr(t))

    ab = SAB(a,b)
    print(repr(ab))



    e = t

    label = "APID=%s BPID=%d G4CXSimtraceMinTest.sh  # A:%s B:%s " % (a.pid, b.pid, a.label, b.label )

    if MODE in [0,1]:
        print("not plotting as MODE %d in environ" % MODE )
    elif MODE == 2:
        pl = mpplt_plotter(label=label)
        fig, axs = pl
        assert len(axs) == 1
        ax = axs[0]

        ax.set_xlim(-356,356)
        ax.set_ylim(-201,201)

    elif MODE == 3:
        pl = pvplt_plotter(label)
        pvplt_viewpoint(pl)   # sensitive to EYE, LOOK, UP envvars
        pvplt_frame(pl, e.f.sframe, local=not GLOBAL )
    pass

    pp =  e.f.simtrace[:,1,:3]  

    gpos = np.ones( [len(pp), 4 ] ) 
    gpos[:,:3] = pp
    lpos = np.dot( gpos, e.f.sframe.w2m )
    upos = gpos if GLOBAL else lpos


    H,V = 0,2  # X, Z

    if SEL == 1:
        sel = np.logical_and( np.abs(upos[:,H]) < 500, np.abs(upos[:,V]) < 500 )
        spos = upos[sel]
    else:
        spos = upos
    pass


    if MODE == 2:
        ax.scatter( spos[:,H], spos[:,V], s=0.1 )
    elif MODE == 3:
        pl.add_points(spos[:,:3])
    pass


    if not a is None and a.pid > -1:
        onephotonplot(pl, a)
    pass 
    if not b is None and b.pid > -1:
        onephotonplot(pl, b)
    pass 


    if MODE == 2:
        fig.show()
    elif MODE == 3:
        pl.show()
    pass
pass

