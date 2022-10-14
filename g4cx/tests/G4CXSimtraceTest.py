#!/usr/bin/env python
"""
G4CXSimtraceTest.py
=====================

TODO: further reduce duplication with cx/tests/CSGOptiXSimtraceTest.py  

Used from gxt.sh, eg::

   SIMPLE=1 ./gxt.sh ana
   ./gxt.sh ana

   MASK=non ./gxt.sh ana
       when the genstep grid is too small the view with default MASK=pos
       will be very zoomed in, use MASK=non to not mask the positions plotted
       by the genstep grid points 

   MASK=t ./gxt.sh ana
       MASK=t requires t>0 which excludes miss, but does not restrict to 
       the genstep grid positions so can see intersects from outside the grid 


"""
import os, numpy as np, logging
log = logging.getLogger(__name__)

from opticks.ana.fold import Fold
from opticks.ana.p import *       
# including cf loaded from CFBASE
# HMM: kinda bad form to instanciate cf from the module load

from opticks.ana.feature import SimtraceFeatures
from opticks.ana.framegensteps import FrameGensteps
from opticks.ana.simtrace_positions import SimtracePositions
from opticks.ana.simtrace_plot import SimtracePlot, pv, mp
from opticks.ana.pvplt import *
from opticks.ana.eget import efloatlist_, elookce_, elook_epsilon_, eint_

from opticks.npy.mortonlib.morton2d import morton2d 




SPURIOUS = "SPURIOUS" in os.environ
RERUN = "RERUN" in os.environ
SELECTION = "SELECTION" in os.environ
SIMPLE = "SIMPLE" in os.environ
MASK = os.environ.get("MASK", "pos")
FEAT = os.environ.get("FEAT", "pid" )
PIDX = eint_("PIDX", "0")


def simple(pl, gs, pos):
    """ 
    :param pl: pvplt_plotter instance
    :param gs: FrameGensteps instance
    :param pos: SimtracePositions instance
    """
    pvplt_simple(pl, gs.centers_local[:,:3], "gs.centers_local[:,:3]" )   
    pvplt_simple(pl, pos.gpos[:,:3], "pos.gpos[:,:3]" )
    pvplt_simple(pl, pos.lpos[:,:3], "pos.lpos[:,:3]" )




def spurious_2d_outliers(bbox, upos):
    """
    :param bbox: array of shape (2,3) eg from t.sframe.bbox
    :param upos: local frame positions expected to be within bbox, eg from t_pos.upos  

    """
    xbox = bbox[1] - bbox[0] 
    fpos = np.zeros( (len(upos), 3), dtype=np.float32 )
    dim = []
    for d in [X,Y,Z]:
        if xbox[d] > 0.:
            dim.append(d)
            fpos[:,d] = (upos[:,d] - bbox[0,d])/xbox[d]
        pass
    pass 
    assert len(dim) == 2, "expecting intersects to be in 2D plane" 
    assert len(np.where(fpos > 1)[0]) == 0, "SPURIOUS running needs all isect within bbox, eg use MASK=t " 
    assert len(np.where(fpos < 0)[0]) == 0, "SPURIOUS running needs all isect within bbox, eg use MASK=t "
    pass
    ipos = np.array( fpos*0xffffffff , dtype=np.uint64 )   ## 32bit range integer coordinates stored in 64 bit  
    kpos = morton2d.Key(ipos[:,dim[0]], ipos[:,dim[1]])    ## morton interleave the two coordinates into one 64 bit code
  

    SPURIOUS_CUT = 1 + int(os.environ.get("SPURIOUS", "1"))
 

    ## scrub low bits and apply uniquing as data reduction : how many bits determines the coarseness   
    u_kpos_0, i_kpos_0, c_kpos_0 = np.unique( kpos & ( 0xfff << 52 ), return_index=True, return_counts=True)   
    sel = c_kpos_0 < SPURIOUS_CUT 

    ## finding outliers in 2d is reduced to finding outliers in sorted list of uint 
    ## and can also use the counts : outliers are expected to be low count 

    u_kpos = u_kpos_0[sel]
    i_kpos = i_kpos_0[sel]  # original upos index of the unique 

    if len(i_kpos) < 10:
        log.info("spurious_2d_outliers SPURIOUS_CUT %d ", SPURIOUS_CUT )
        log.info("i_kpos\n%s" % str(i_kpos))
        log.info("upos[i_kpos]\n%s" % str(upos[i_kpos]) )
    pass                                                                                                                                                                                

    ## decode the selected isolated morton codes to find the probable spurious intersects
    d_kpos = morton2d.Decode(u_kpos)  ## HMM returns tuple of 2 which is bit inconvenient  
    t_spos = np.zeros( (len(u_kpos),3), dtype=np.float32 )
    for idim,d in enumerate(dim):
        t_spos[:,d] = d_kpos[idim].astype(np.float32)/np.float32(0xffffffff)
        t_spos[:,d] *= xbox[d]
        t_spos[:,d] += bbox[0,d]
    pass 
    return u_kpos, c_kpos_0, i_kpos, t_spos




if __name__ == '__main__':

    #fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    fmt = '{%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)

    t = Fold.Load(symbol="t")
    a = Fold.Load("$A_FOLD", symbol="a")
    b = Fold.Load("$B_FOLD", symbol="b")
    print("cf.cfbase : %s " % cf.cfbase) 

    print("---------Fold.Load.done")
    x = a 

    print(repr(t))
    print(repr(a))
    print(repr(b))

    print("---------print.done")


    if not a is None and not a.seq is None:
        a_nib = seqnib_(a.seq[:,0])                  # valid steppoint records from seqhis count_nibbles
        a_gpos_ = a.record[PIDX,:a_nib[PIDX],0,:3]   # global frame photon step record positions of single PIDX photon
        a_gpos  = np.ones( (len(a_gpos_), 4 ) )
        a_gpos[:,:3] = a_gpos_
        a_lpos = np.dot( a_gpos, t.sframe.w2m )      # a global positions into gxt target frame 
    else:
        a_lpos = None
    pass

    if not b is None and not b.seq is None:
        b_nib = seqnib_(b.seq[:,0])                  # valid steppoint records from seqhis count_nibbles
        b_gpos_ = b.record[PIDX,:b_nib[PIDX],0,:3]  # global frame photon step record positions of single PIDX photon
        b_gpos  = np.ones( (len(b_gpos_), 4 ) )
        b_gpos[:,:3] = b_gpos_
        b_lpos = np.dot( b_gpos, t.sframe.w2m ) 
    else:
        b_lpos = None
    pass

    x_lpos = a_lpos




    local = True 

    t_gs = FrameGensteps(t.genstep, t.sframe, local=local, symbol="gs" )  ## get gs positions in target frame
    print(t_gs)


    if RERUN:
        log.info("RERUN envvar switched on use of simtrace_rerun from CSG/SimtraceRerunTest.sh " ) 
        simtrace = t.simtrace_rerun
    else:
        simtrace = t.simtrace
    pass

    t_pos = SimtracePositions(simtrace, t_gs, t.sframe, local=local, mask=MASK, symbol="t_pos" )
    print(t_pos)


    if SPURIOUS:
        log.info("SPURIOUS envvar switches on search for morton outliers using spurious_2d_outliers ")
        u_kpos, c_kpos, i_kpos, t_spos = spurious_2d_outliers( t.sframe.bbox, t_pos.upos )
        j_kpos = t_pos.upos2simtrace[i_kpos]
        log.info("j_kpos = t_pos.upos2simtrace[i_kpos]\n%s" % str(t_pos.upos2simtrace[i_kpos]) )
        log.info("simtrace[j_kpos]\n%s" % str(simtrace[j_kpos]) )
        simtrace_spurious = j_kpos
    else:
        t_spos = None
        simtrace_spurious = []
    pass

    if SIMPLE:
        pl = pvplt_plotter()
        simple(pl, t_gs, t_pos)
        pl.show()
        raise Exception("SIMPLE done")
    pass

    pf = SimtraceFeatures(t_pos, cf, featname=FEAT, symbol="pf" )

    pl = SimtracePlot.MakePVPlotter()

    plt = SimtracePlot(pl, pf.feat, t_gs, t.sframe, t_pos, outdir=os.path.join(t.base, "figs") )


    if not x_lpos is None:           
        plt.x_lpos = x_lpos   
    pass
    if not t_spos is None:           
        plt.t_spos = t_spos     ## spurious intersects identified by morton2d isolation 
    pass

    ## created by CSG/SimtraceRerunTest.sh with SELECTION envvar picking simtrace indices to highlight 
    ## but the SELECTION envvar used here just has to exist to trigger selection plotting 
    if hasattr(t, "simtrace_selection") and SELECTION:  
        plt.simtrace_selection = t.simtrace_selection
    elif len(simtrace_spurious) > 0:
        plt.simtrace_selection = simtrace[simtrace_spurious]
    else:
        pass
    pass 
    
    if not mp is None:
        plt.positions_mpplt()
        ax = plt.ax
    pass

    if not pv is None:
        plt.positions_pvplt()
    pass
pass


