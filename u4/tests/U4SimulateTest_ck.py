#!/usr/bin/env python
"""
U4SimulateTest_ck.py
======================

"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.fold import Fold
from opticks.ana.p import * 
from opticks.sysrap.sevt import SEvt

MODE = int(os.environ.get("MODE","2"))
if MODE > 0:
    from opticks.ana.pvplt import * 
else:
    pass
pass

axes = 0, 2  # X,Z
H,V = axes 


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    FOLD = os.environ.get("FOLD", None)
    log.info(" -- SEvt.Load FOLD" )
    a = SEvt.Load(FOLD, symbol="a")   # optional photon histories 
    print(a)

    beg_ = "a.f.record[:,0,0,:3]"
    beg = eval(beg_)

    end_ = "a.f.photon[:,0,:3]"
    end = eval(end_)


    #label0, ppos0 = None, None
    label0, ppos0 = "b:%s" % beg_ , beg

    #label0, ppos0 = None, None
    label1, ppos1 = "r:%s" % end_ , end


    HEADLINE = "%s %s" % ( a.LAYOUT, a.CHECK )
    label = "\n".join( filter(None, [HEADLINE, label0, label1]))
    print(label)

    if MODE == 0:
        print("not plotting as MODE 0  in environ")
    elif MODE == 2:
        fig, ax = mpplt_plotter(label=label)

        ax.set_ylim(-250,250)
        ax.set_xlim(-500,500)

        if not ppos0 is None: ax.scatter( ppos0[:,H], ppos0[:,V], s=1, c="b" )  
        if not ppos1 is None: ax.scatter( ppos1[:,H], ppos1[:,V], s=1, c="r" )  

        fig.show()
    elif MODE == 3:
        pl = pvplt_plotter(label)
        os.environ["EYE"] = "0,100,165"
        os.environ["LOOK"] = "0,0,165"
        pvplt_viewpoint(pl)
        pl.add_points(ppos0)
        pl.show()
    pass
pass

