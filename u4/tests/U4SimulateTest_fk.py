#!/usr/bin/env python

import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.fold import Fold, RFold
from opticks.ana.p import * 
from opticks.u4.tests.U4SimulateTest import U4SimulateTest

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
    log.info(" -- U4SimulateTest.Load FOLD" )
    a = U4SimulateTest.Load(FOLD, symbol="a")   # optional photon histories 
    print(a)

    SPECS = np.array(a.f.U4R_names.lines)
    st_ = a.f.aux[:,:,2,3].view(np.int32)
    st = SPECS[st_]

 
    w_fk_ = "np.where(a.fk>0)"
    w_fk = eval(w_fk_)
    print(w_fk_,"\n",w_fk) 

    w_fk8_ = "np.where(a.fk==8)"
    w_fk8 = eval(w_fk8_)
    print(w_fk8_,"\n",w_fk8) 


    tab_fk_ = "np.c_[np.unique(a.fk,return_counts=True)]"  
    tab_fk  = eval(tab_fk_)
    print(tab_fk_,"\n",tab_fk) 
     
    qqtab_fk_ =  "np.c_[np.unique(a.qq[w_fk], return_counts=True)]"
    qqtab_fk = eval(qqtab_fk_)  
    print(qqtab_fk_,"\n",qqtab_fk) 

    post_fk_ = "a.f.record[tuple(w_fk[0]),tuple(w_fk[1]),0]"
    post_fk = eval(post_fk_)  
    print(post_fk_,"\n",post_fk) 

    post_fk8_ = "a.f.record[tuple(w_fk8[0]),tuple(w_fk8[1]),0] # FAKE_MANUAL all on midline "
    post_fk8 = eval(post_fk8_)  
    print(post_fk8_,"\n",post_fk8) 

    tab_st_fk_ = "np.c_[np.unique(st[w_fk],return_counts=True)]"
    tab_st_fk = eval(tab_st_fk_)
    print(tab_st_fk_,"\n",tab_st_fk) 

    tab_st_fk8_ = "np.c_[np.unique(st[w_fk8],return_counts=True)]"
    tab_st_fk8 = eval(tab_st_fk8_)
    print(tab_st_fk8_,"\n",tab_st_fk8_) 
  


    #label0, ppos0 = None, None
    label0, ppos0 = "b:%s" % post_fk_ , post_fk[:,:3]

    #label1, ppos1 = None, None
    label1, ppos1 = "r:%s" % post_fk8_ ,post_fk8


    HEADLINE = "%s %s" % ( a.LAYOUT, a.CHECK )
    label = "\n".join( filter(None, [HEADLINE, label0, label1]))
    print(label)

    if MODE == 0:
        print("not plotting as MODE 0  in environ")
    elif MODE == 2:
        fig, axs = mpplt_plotter(label=label)
        assert len(axs) == 1 
        ax = axs[0]

        ax.set_ylim(-250,250)
        ax.set_xlim(-500,500)

        if not ppos0 is None: ax.scatter( ppos0[:,H], ppos0[:,V], s=1 )  
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
