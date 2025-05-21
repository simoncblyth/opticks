#!/usr/bin/env python
"""
SABTest.py : analysis script used from SABTest.sh
==================================================

Started from ~/j/InputPhotonsCheck/InputPhotonsCheck.py

"""

import os, logging, numpy as np
log = logging.getLogger(__name__)
os.environ["MODE"] = "3" 

from opticks.sysrap.sevt import SEvt, SAB, SABHit
from opticks.ana.p import cf

MODE = int(os.environ.get("MODE","3"))
MODE0 = MODE
print("SABTest.py initial MODE:%d " % MODE)


GLOBAL = int(os.environ.get("GLOBAL","0")) == 1
PICK = os.environ.get("PICK","A")
INCPOI = float(os.environ.get("INCPOI","0"))   ## increment pyvista presentation point and line size, often need -5 


if MODE in [2,3]:
    import opticks.ana.pvplt as pvp
    print(pvp.__doc__)
    print("after import MODE:%d " % MODE)
    MODE = MODE0
    # HMM this import overrides MODE, so need to keep defaults the same
pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    INST_FRAME = int(os.environ.get("INST_FRAME","-1"))
    if INST_FRAME == -1: 
        print("INST_FRAME %d : using default sframe saved with the sevt" % INST_FRAME )
        M2W_OVERRIDE = None
        W2M_OVERRIDE = None
    else:
        print("INST_FRAME %d : W2M_OVERRIDE obtained from cf.inst controlled by envvar INST_FRAME " % INST_FRAME )
        M2W_OVERRIDE = np.eye(4)
        M2W_OVERRIDE[:,:3] = cf.inst[INST_FRAME][:,:3]
        W2M_OVERRIDE = np.linalg.inv(M2W_OVERRIDE)
    pass


    print(__doc__)

    a = SEvt.Load("$AFOLD", symbol="a", W2M=W2M_OVERRIDE)
    b = SEvt.Load("$BFOLD", symbol="b", W2M=W2M_OVERRIDE)
    print(repr(a))
    print(repr(b))


    if not a is None and b is None:
        PICK = "A"
    elif not b is None and a is None:
        PICK = "B"
    else:
        pass
    pass

    if not a is None and not b is None:
        if "SAB" in os.environ:
            print("[--- ab = SAB(a,b) ----")
            ab = SAB(a,b)
            print("]--- ab = SAB(a,b) ----")

            print("[----- repr(ab) ")
            print(repr(ab))
            print("]----- repr(ab) ")
        else:
            print(" NOT doing ab = SAB(a,b) # do that with : export SAB=1" )
        pass

        print("[--- abh = SABHit(a,b) ----")
        abh = SABHit(a,b)
        print("]--- abh = SABHit(a,b) ----")
        print("[----- repr(abh) ")
        print(repr(abh))
        print("]----- repr(abh) ")
    pass

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

    context = "PICK=%s MODE=%d  ~/j/jtds/jtds.sh " % (PICK, MODE )
    print(context)

    pl = None
    for e in ee:
        if e is None:continue
        elabel = "%s : %s " % ( e.symbol.upper(), e.f.base )

        if hasattr(e.f, 'photon'):
            pos = e.f.photon[:,0,:3]
        elif hasattr(e.f, 'hit'):
            pos = e.f.hit[:,0,:3]
        else:
            log.info("%s:sevt lacks photon or hit" % e.symbol)
            pos = None
        pass

        if hasattr(e.f, 'record'):
            sel = np.where(e.f.record[:,:,2,3] > 0) # select on wavelength to avoid unfilled zeros
            poi = e.f.record[:,:,0,:3][sel]
        else:
            log.info("%s:sevt lacks record" % e.symbol)
            poi = None
        pass

        if hasattr(e.f, 'sframe'):
            _W2M = e.f.sframe.w2m
            log.info("%s:sevt has sframe giving _W2M " % e.symbol)
        else:
            _W2M = np.eye(4)
            log.info("%s:sevt lacks sframe" % e.symbol)
        pass

        W2M = _W2M if W2M_OVERRIDE is None else W2M_OVERRIDE


        print("W2M_OVERRIDE (from manual INST_FRAME, no longer typical)\n",W2M_OVERRIDE)
        print("_W2M(from %s.f.sframe.w2m, now standard)\n" % e.symbol,_W2M)
        print("W2M\n",W2M)

        if not pos is None:
            print("transform gpos (from photon or hit array) to lpos using W2M : ie into target frame : GLOBAL %d " % GLOBAL)
            gpos = np.ones( [len(pos), 4 ] )
            gpos[:,:3] = pos
            lpos = np.dot( gpos, W2M )   # hmm unfilled global zeros getting transformed somewhere
            upos = gpos if GLOBAL else lpos
        else:
            upos = None
        pass

        if not poi is None:
            print("transform gpoi (from record array) to lpoi using W2M : ie into target frame : GLOBAL %d " % GLOBAL)
            gpoi = np.ones( [len(poi), 4 ] )
            gpoi[:,:3] = poi
            lpoi = np.dot( gpoi, W2M )
            upoi = gpoi if GLOBAL else lpoi
        else:
            upoi = None
        pass


        label = context + " ## " + elabel

        if MODE == 3 and not "SAB" in os.environ:
            if pvp.pv is None:
                print("MODE is 3 but pvp.pv is None, get into ana environment first eg with hookup_conda_ok")
            else:
                pl = pvp.pvplt_plotter(label)
                pvp.pvplt_viewpoint(pl) # sensitive EYE, LOOK, UP, ZOOM envvars eg EYE=0,-3,0

                if not upoi is None:
                    print("add_points green for the upoi : record points ")
                    pl.add_points( upoi[:,:3], color="green", point_size=3.0 )
                pass
                if not upos is None:
                    print("add_points red  for the upos : photon/hit points ")
                    pl.add_points( upos[:,:3], color="red", point_size=3.0 )
                pass
                pl.show_grid()
                pl.increment_point_size_and_line_width(INCPOI)
                cp = pl.show()
            pass
        pass
    pass



pass


