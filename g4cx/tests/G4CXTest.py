#!/usr/bin/env python 

import os, logging, numpy as np
from opticks.ana.fold import Fold, IsRemoteSession
from opticks.sysrap.sevt import SEvt, SAB
from opticks.ana.p import cf

log = logging.getLogger(__name__)

GLOBAL = int(os.environ.get("GLOBAL","0")) == 1
MODE = int(os.environ.get("MODE","3")) 
SEL = int(os.environ.get("SEL","0")) 

PICK = os.environ.get("PICK","AB")
AIDX = int(os.environ.get("AIDX","0"))
BIDX = int(os.environ.get("BIDX","0"))



if IsRemoteSession():  # HMM: maybe do this inside pvplt ?
    MODE = 0
    print("detect fold.IsRemoteSession forcing MODE:%d" % MODE)
elif MODE in [2,3]:
    from opticks.ana.pvplt import *  # HMM this import overrides MODE, so need to keep defaults the same 
pass




def onephotonplot(pl, e ):
    """
    :param pl: MODE 2/2 plotter objects
    :param e: SEvt instance
    """
    if e is None: return
    if e.pid < 0: return

    print("onephotonplot e.pid %d " % e.pid )

    if not hasattr(e,'l'): return
    if e.l is None: return
    print("onephotonplot e.pid %d PROCEED " % e.pid )

    rpos =  e.l[:,:3] + e.off

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
    print("GLOBAL:%d MODE:%d SEL:%d" % (GLOBAL,MODE, SEL))

    a = SEvt.Load("$AFOLD", symbol="a")
    print(repr(a))
    b = SEvt.Load("$BFOLD", symbol="b")
    print(repr(b))


    print(cf)


    #sli="[:]"  allowing everything makes for big tables of low stat histories
    sli="[:15]"
    at = a.minimal_qtab(sli=sli)  
    bt = b.minimal_qtab(sli=sli)

    print("at\n",at)
    print("bt\n",bt)

    ab = SAB(a,b)
    print(repr(ab))


    assert PICK in ["A","B","AB","BA", "CF"]
    if PICK == "A":
        ee = [a,] 
    elif PICK == "B":
        ee = [b,]
    elif PICK == "AB":
        ee = [a,b,]
    elif PICK == "BA":
        ee = [b,a,]
    elif PICK == "CF":
        ee = []
    pass 

    context = "PICK=%s MODE=%d SEL=%d ./G4CXAppTest.sh ana " % (PICK, MODE, SEL )
    print(context)

    #hsel = None
    hsel = "TO BT BT BT SD,TO BT BT BT SA"
    HSEL = os.environ.get("HSEL", hsel)


    for e in ee:
        ew = e.q_startswith_or_(HSEL) if not HSEL is None else None

        elabel = "%s : %s " % ( e.symbol.upper(), e.f.base )
        if not ew is None:
            elabel += " HSEL=%s " % HSEL 
        pass
        label = context + " ## " + elabel

        qtab = e.minimal_qtab()


        if MODE in [0,1]:
            print("not plotting as MODE %d in environ" % MODE )
        elif MODE == 2:
            pl = mpplt_plotter(label=label)
            fig, axs = pl
            assert len(axs) == 1
            ax = axs[0]

            xlim, ylim = mpplt_focus_aspect()
            if not xlim is None:
                ax.set_xlim(xlim) 
                ax.set_ylim(ylim) 
            else:
                log.info("mpplt_focus_aspect not enabled, use eg FOCUS=0,0,100 to enable ")
            pass 

            #ax.set_xlim(-356,356)
            #ax.set_ylim(-201,201)

        elif MODE == 3:
            pl = pvplt_plotter(label)
            pvplt_viewpoint(pl)   # sensitive to EYE, LOOK, UP envvars
            pvplt_frame(pl, e.f.sframe, local=not GLOBAL )
        pass

        #pp = e.f.inphoton[:,0,:3]
        #pp = e.f.photon[:,0,:3]
        pp = e.f.record[:,:,0,:3]
        ww = pp[ew] if not ew is None else None

        ppf = pp.reshape(-1,3)
        wwf = ww.reshape(-1,3) if not ww is None else None
        

        g_pos = np.ones( [len(ppf), 4 ] ) 
        g_pos[:,:3] = ppf
        l_pos = np.dot( g_pos, e.f.sframe.w2m )
        u_pos = g_pos if GLOBAL else l_pos


        if not wwf is None: 
            h_pos = np.ones( [len(wwf), 4 ] ) 
            h_pos[:,:3] = wwf
            i_pos = np.dot( h_pos, e.f.sframe.w2m )
            v_pos = h_pos if GLOBAL else i_pos 
        else:
            h_pos = None
            i_pos = None
            v_pos = None
        pass


        H,V = 0,2  # X, Z

        if SEL == 1:
            sel = np.logical_and( np.abs(u_pos[:,H]) < 500, np.abs(u_pos[:,V]) < 500 )
            s_pos = u_pos[sel]
        else:
            s_pos = u_pos
        pass


        if MODE == 2:
            ax.scatter( s_pos[:,H], s_pos[:,V], s=0.1 )
        elif MODE == 3:
            pl.add_points(s_pos[:,:3])
        else:
            pass
        pass


        if not v_pos is None:
            if MODE == 2:
                ax.scatter( v_pos[:,H], v_pos[:,V], s=0.1, c="r" )
            elif MODE == 3:
                pl.add_points(v_pos[:,:3], color="red")
            else:
                pass
            pass
        else:
            pass
        pass






        if e.pid > -1: 
            onephotonplot(pl, e)
        pass 



        if MODE == 2:
            fig.show()
        elif MODE == 3:
            pl.show()
        pass
    pass    ## ee loop 
pass
