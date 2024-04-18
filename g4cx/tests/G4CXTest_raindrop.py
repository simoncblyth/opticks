#!/usr/bin/env python
"""
G4CXTest_raindrop.py
======================


"""
import os, logging, textwrap, numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.sevt import SEvt, SAB
from opticks.ana.nbase import chi2_pvalue

MODE = int(os.environ.get("MODE","3"))
PICK = os.environ.get("PICK","B")

if MODE in [2,3]:
    try:
        import pyvista as pv
        from opticks.ana.pvplt import pvplt_plotter, pvplt_viewpoint
    except ImportError:
        pv = None
    pass
else:
    pv = None
pass


def eprint(expr, l, g ):
    print(expr)
    try:
       val = eval(expr,l,g)
    except AttributeError:
       val = "eprint:AttributeError"
    pass
    print(val)
pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    a = SEvt.Load("$AFOLD", symbol="a")
    print(repr(a))
    b = SEvt.Load("$BFOLD", symbol="b")
    print(repr(b))
    ee = {'A':a, 'B':b} 

    print("[--- ab = SAB(a,b) ----")

    if not a is None and not b is None:
        ab = SAB(a,b)
    else:
        ab = None
    pass 
    print("]--- ab = SAB(a,b) ----")

    print("[----- repr(ab) ")
    print(repr(ab))
    print("]----- repr(ab) ")


    EXPR_ = r"""
    np.c_[np.unique(a.q, return_counts=True)] 
    np.c_[np.unique(b.q, return_counts=True)] 
    """
    EXPR = list(filter(None,textwrap.dedent(EXPR_).split("\n")))
    for expr in EXPR:eprint(expr, locals(), globals() )


    #select = "WL"            ## select on wavelength to avoid unfilled zeros
    #select = "TO BT BT SA"   ## select on photon history
    select = "TO BT SA,TO BR BT SA" 
    MSELECT = os.environ.get("SELECT", select )  

    sli = slice(0,100000)   # restrict to 1M to stay interactive

    speed_ = lambda r,i:np.sqrt(np.sum( (r[:,i+1,0,:3]-r[:,i,0,:3])*(r[:,i+1,0,:3]-r[:,i,0,:3]), axis=1))/(r[:,i+1,0,3]-r[:,i,0,3])

    for Q in PICK:
        e = ee.get(Q,None)
        if e is None:continue

        for SELECT in MSELECT.split(","):
            context = "PICK=%s MODE=%d SELECT=\"%s\" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh " % (Q, MODE, SELECT )
            print(context)

            if SELECT == "WL": 
                sel = np.where(e.f.record[:,:,2,3] > 0) 
                sel_p, sel_r = sel  # 2-tuple selecting photon and record points   
                sel_p = sel_p[sli]
                sel_r = sel_r[sli]
            else:
                sel_p = e.q_startswith(SELECT)  
                sel_p = sel_p[sli]
            
                SELECT_ELEM = SELECT.split()
                SELECT_POINT = len(SELECT_ELEM)
            
                sel_r = slice(0,SELECT_POINT)
                r = e.f.record[sel_p,sel_r]  

                if "SAVE_SEL" in os.environ:
                    sel_dir = os.path.join(e.f.base, SELECT.replace(" ", "_") )
                    if not os.path.isdir(sel_dir):
                        os.makedirs(sel_dir)
                    pass
                    sel_path = os.path.join( sel_dir, "record.npy" )
                    print("REC=%s ~/o/examples/UseGeometryShader/run.sh" % sel_dir) 
                    np.save( sel_path, e.f.record[sel_p] )
                pass

                for i in range(SELECT_POINT-1):
                    speed = speed_(r,i)
                    speed_min = speed.min() if len(speed) > 0 else -1 
                    speed_max = speed.max() if len(speed) > 0 else -1 
                    fmt = "speed len/min/max for : %d -> %d : %s -> %s : %8d %7.3f %7.3f "
                    print(fmt % (i,i+1,SELECT_ELEM[i],SELECT_ELEM[i+1],len(speed),speed_min, speed_max ))
                pass

                if Q == "B": 
                    kludge = e.f.NPFold_meta.U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE
                    print("e.f.NPFold_meta.U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:%s " % kludge )
                    print("e.f.NPFold_meta.G4VERSION_NUMBER:%s " % e.f.NPFold_meta.G4VERSION_NUMBER )
                pass
            pass

            _pos = e.f.photon[sel_p,0,:3]        ## end position 
 
            _beg = e.f.record[sel_p,0,0,:3]      ## begin positions 
            _poi = e.f.record[sel_p,sel_r,0,:3]  ## all positions 

            print("_pos.shape %s " % str(_pos.shape))
            print("_beg.shape %s " % str(_beg.shape))
            print("_poi.shape %s " % str(_poi.shape))

            pos = _pos.reshape(-1,3)
            beg = _beg.reshape(-1,3)
            poi = _poi.reshape(-1,3)

            elabel = "%s : %s " % ( e.symbol.upper(), e.f.base )
            label = context + " ## " + elabel

            if MODE == 3 and not pv is None:
                if len(poi) == 0:
                    print("FAILED TO SELECT ANY POI : SKIP PLOTTING" )
                else:
                    pl = pvplt_plotter(label)
                    pvplt_viewpoint(pl) # sensitive EYE, LOOK, UP, ZOOM envvars eg EYE=0,-3,0 
                    pl.add_points( poi, color="green", point_size=3.0 )
                    pl.add_points( pos, color="red", point_size=3.0 )
                    pl.add_points( beg, color="blue", point_size=3.0 )
                    if not "NOGRID" in os.environ: pl.show_grid()
                    cp = pl.show()
                pass
            pass
        pass
    pass
        

