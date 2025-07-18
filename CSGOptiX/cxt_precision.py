#!/usr/bin/env python
"""
cxt_precision.py
==================


"""
import os, sys, re, logging, textwrap, numpy as np
np.set_printoptions(linewidth=200,edgeitems=20)

log = logging.getLogger(__name__)

from opticks.ana.fold import Fold
from opticks.sysrap.sevt import SEvt
import matplotlib.pyplot as mp
SIZE=np.array([1280, 720])


if __name__ == '__main__':
    r = SEvt.Load("$RFOLD", symbol="r")
    a = SEvt.Load("$AFOLD", symbol="a")

    RFOLD = os.environ["RFOLD"]
    AFOLD = os.environ["AFOLD"]

    FIGPATH = os.environ.get("FIGPATH", None)

    iprs = os.environ["OPTICKS_INPUT_PHOTON_RECORD_SLICE"]
    iprt = os.environ["OPTICKS_INPUT_PHOTON_RECORD_TIME"]
    opr = int(os.environ.get("OPTICKS_PROPAGATE_REFINE", "0"))
    oprd = float(os.environ.get("OPTICKS_PROPAGATE_REFINE_DISTANCE", "-1"))

    _title = [
               "TEST=%s ~/opticks/CSGOptiX/cxt_precision.sh" % os.environ["TEST"],
               "OPTICKS_INPUT_PHOTON_RECORD_SLICE %s" % iprs,
               "OPTICKS_INPUT_PHOTON_RECORD_TIME %s" % iprt ,
               "OPTICKS_PROPAGATE_REFINE %s OPTICKS_PROPAGATE_REFINE_DISTANCE %s" % (opr,oprd)
             ]
    title = "\n".join(_title)

    rw = r.q__(iprs)
    tt = np.arange(*list(map(float,iprt[1:-1].split(":"))))


    _nt = "a.f.simtrace.shape[0]//rw.shape[0]"
    nt = eval(_nt)

    _s = "a.f.simtrace.reshape(-1,nt,4,4)"
    s = eval(_s)

    _e = "np.linalg.norm((s[:,:,1,:3] - s[:,-1,1,:3][:,np.newaxis]).reshape(-1,3),axis=1).reshape(-1,nt)[:,:-1]"
    e = eval(_e)

    _d = "s[:,:,0,3][:,:-1]"
    d = eval(_d)

    _dav = "np.average(d, axis=0)"
    dav = eval(_dav)

    _eav = "np.average(e, axis=0)"
    eav = eval(_eav)

    _emx = "np.max(e, axis=0 )"
    emx = eval(_emx)

    _emn = "np.min(e, axis=0 )"
    emn = eval(_emn)

    _emd = "np.median(e, axis=0 )"
    emd = eval(_emd)


    EXPR = r"""
    title

    RFOLD
    r

    AFOLD  # special simtrace precision run
    a

    iprs   # OPTICKS_INPUT_PHOTON_RECORD_SLICE

    iprt   # OPTICKS_INPUT_PHOTON_RECORD_TIME
    tt.shape
    tt

    a.f.simtrace.shape
    rw.shape
    a.f.simtrace.shape[0]//rw.shape[0]     # nt
    a.f.simtrace.shape[0] % rw.shape[0]    # expect zero

    _s
    #s
    s.shape

    _nt
    nt

    len(tt) == nt

    # distance from intersect positions to last(and most precise) intersect position
    _e
    e
    e.shape

    # distance from origin to intersect
    _d
    d
    d.shape

    _dav   # average over clumps
    dav

    _eav   # average over clumps
    eav

    _emx
    emx

    _emn
    emn

    _emd
    emd

    """
    for expr in list(map(str.strip, textwrap.dedent(EXPR).split("\n"))):
        print(expr)
        if expr == "" or expr[0] == "#": continue
        print(repr(eval(expr)))
    pass




    fig, ax = mp.subplots(figsize=SIZE/100.)
    fig.suptitle(title)


    ax.scatter( d.ravel(), e.ravel(), s=0.2 )

    #ax.plot( dav, eav, label="avg" )
    #ax.plot( dav, emn, label="min" )

    ax.plot( dav, emx, label="max" )
    ax.plot( dav, emd, label="median" )

    ax.axhline(0.05, linestyle='--', label="EPSILON (0.05 mm)")

    ax.set_ylabel("Intersect Error (mm)")
    ax.set_xlabel("Ray trace distance (mm)")

    #ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_ylim( 4e-5, 1.1 )

    ax.legend()

    fig.show()

    if not FIGPATH is None:
        print("savefig to FIGPATH:%s" % FIGPATH)
        mp.savefig(FIGPATH)
    else:
        print("no-savefig as on FIGPATH")
    pass




