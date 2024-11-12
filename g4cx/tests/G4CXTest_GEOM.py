#!/usr/bin/env python
"""
G4CXTest_GEOM.py
======================

This script is intended for simple plotting and 
statistical SEvt comparison. 

The alternate script G4CXTest.py is used for 
more detailed plotting such as single photon path 
illustration. 

"""
import os, logging, textwrap, numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.sevt import SEvt, SAB
from opticks.ana.nbase import chi2_pvalue

from opticks.ana.p import *  # including cf boundary___


MODE = int(os.environ.get("MODE","3"))
SEL = int(os.environ.get("SEL","0"))
HSEL = os.environ.get("HSEL","")    # History selection 
PICK = os.environ.get("PICK","A")
SAB_ = "SAB" in os.environ

H,V = 0,2  # X, Z



if MODE in [2,3]:
    try:
        import pyvista as pv
        from opticks.ana.pvplt import pvplt_plotter, pvplt_viewpoint, mpplt_plotter, mpplt_focus_aspect
    except ImportError:
        pv = None
    pass
else:
    pv = None
pass


def eprint(expr, _locals, _globals ):
    print(expr)
    try:
       val = eval(expr,_locals,_globals)
    except AttributeError:
       val = "eprint:AttributeError"
    pass
    print(val)
pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("MODE:%s SEL:%s PICK:%s SAB_:%s " % (MODE, SEL, PICK, SAB_) ) 

    if PICK == "A" or PICK == "AB" or SAB_:
        a = SEvt.Load("$AFOLD", symbol="a")
        print(repr(a))
    else: 
        a = None
    pass

    if PICK == "B" or PICK == "AB" or SAB_:
        b = SEvt.Load("$BFOLD", symbol="b")
        print(repr(b))
    else:
        b = None
    pass


    if not HSEL == "":
        print("doing HSEL [%s] history selection " % HSEL )

        wa = a.q_startswith_or(HSEL) if not a is None else None
        wa_shape = "-" if wa is None else repr(wa.shape)
        print("wa.shape %s " % wa_shape)

        wb = b.q_startswith_or(HSEL) if not b is None else None
        wb_shape = "-" if wb is None else repr(wb.shape)
        print("wb.shape %s " % wb_shape)

        ra = a.f.record[wa] if not a is None else None
        ra_shape = "-" if ra is None else repr(ra.shape)
        print("ra.shape %s " % ra_shape)

        rb = b.f.record[wb] if not b is None else None
        rb_shape = "-" if rb is None else repr(rb.shape)
        print("rb.shape %s " % rb_shape)





        st = slice(0,4)
        ba = ra[:,st,3,0].view(np.uint32) >> 16 if not ra is None else None
        bb = rb[:,st,3,0].view(np.uint32) >> 16 if not rb is None else None 
        assert np.all( bb == 0 ) # unfortunately no boundary info for B, YET  


        rra = np.sqrt(np.sum(ra[:,st,0,:3]*ra[:,st,0,:3],axis=2)) if not ra is None else None
        rra_shape = "-" if rra is None else repr(rra.shape)  
        print("rra.shape %s " % rra_shape )

        rrb = np.sqrt(np.sum(rb[:,st,0,:3]*rb[:,st,0,:3],axis=2)) if not rb is None else None
        rrb_shape = "-" if rrb is None else repr(rrb.shape)  
        print("rrb.shape %s " % rrb_shape )

    else:
        print("set HSEL to make history selection ")
        pass
    pass 
   
    ee = {'A':a, 'B':b} 

    if SAB_: 
        print("[--- ab = SAB(a,b) ----")
        ab = None if a is None or b is None else SAB(a,b) 
        print("]--- ab = SAB(a,b) ----")

        print("[----- repr(ab) ")
        print(repr(ab))
        print("]----- repr(ab) ")
    else:
        print("ab = SAB(a,b) ## skipped as SAB not in os.environ")
    pass


    # EXPR_ = r"""
    # np.c_[np.unique(a.q, return_counts=True)] 
    # np.c_[np.unique(b.q, return_counts=True)] 
    # """
    # EXPR = list(filter(None,textwrap.dedent(EXPR_).split("\n")))
    # for expr in EXPR:eprint(expr, locals(), globals() )

    context = "PICK=%s MODE=%d  ~/opticks/g4cx/tests/G4CXTest_GEOM.sh " % (PICK, MODE )
    print(context)

    HITONLY = "HITONLY" in os.environ

    for Q in PICK:
        e = ee.get(Q,None)
        if e is None:continue

        elabel = "%s : %s " % ( e.symbol.upper(), e.f.base )
        label = context + " ## " + elabel


        if hasattr(e.f, 'hit'): 
            hit = e.f.hit[:,0,:3]
        else:
            hit = None
        pass

        if hasattr(e.f, 'photon') and not HITONLY: 
            pos = e.f.photon[:,0,:3]
        else:
            pos = None
        pass

        if hasattr(e.f, 'record') and not HITONLY:
            sel = np.where(e.f.record[:,:,2,3] > 0) # select on wavelength to avoid unfilled zeros
            poi = e.f.record[:,:,0,:3][sel]
        else:
            poi = None
        pass

        if MODE == 2:
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

            if SEL == 1:
                #sel = np.logical_and( np.abs(u_pos[:,H]) < 500, np.abs(u_pos[:,V]) < 500 )
                sel = pos[:,V] > 28000

                u_pos = pos[sel]
            else:
                u_pos = pos
            pass

            ax.scatter( u_pos[:,H], u_pos[:,V], s=0.1 )

            fig.show()

        elif MODE == 3 and not pv is None:
            pl = pvplt_plotter(label)
            pvplt_viewpoint(pl) # sensitive EYE, LOOK, UP, ZOOM envvars eg EYE=0,-3,0 

            if not poi is None:
                pl.add_points( poi, color="green", point_size=3.0 )
            pass
            if not pos is None:
                pl.add_points( pos, color="red", point_size=3.0 )
            pass
            if not hit is None:
                pl.add_points( hit, color="cyan", point_size=3.0 )
            pass

            if not "NOGRID" in os.environ:
                pl.show_grid()
            pass
            cp = pl.show()
        pass
    pass
        

