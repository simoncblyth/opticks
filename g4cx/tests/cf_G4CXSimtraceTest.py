#!/usr/bin/env python
"""
cf_G4CXSimtraceTest.py
==============================

"""

import os, numpy as np, logging
log = logging.getLogger(__name__)
from opticks.ana.fold import Fold
from opticks.CSG.Values import Values 
from opticks.ana.eget import efloatarray_, efloatlist_
from opticks.sysrap.sframe import sframe , X, Y, Z
from opticks.ana.framegensteps import FrameGensteps

import matplotlib.pyplot as mp

SIZE = np.array([1280, 720]) 


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    s_geom = os.environ.get("S_GEOM", None)
    t_geom = os.environ.get("T_GEOM", None)
    u_geom = os.environ.get("U_GEOM", None)
    v_geom = os.environ.get("V_GEOM", None)

    s = Fold.Load("$S_FOLD", symbol="s") if not s_geom is None else None
    t = Fold.Load("$T_FOLD", symbol="t") if not t_geom is None else None
    u = Fold.Load("$U_FOLD", symbol="u") if not u_geom is None else None
    v = Fold.Load("$V_FOLD", symbol="v") if not v_geom is None else None

    print(repr(s))
    print(repr(t))
    print(repr(u))
    print(repr(v))

    sv = Values.Find("$S_FOLD", symbol="sv") if not s_geom is None else None
    tv = Values.Find("$T_FOLD", symbol="tv") if not t_geom is None else None
    uv = Values.Find("$U_FOLD", symbol="uv") if not u_geom is None else None
    vv = Values.Find("$V_FOLD", symbol="vv") if not v_geom is None else None

    print(repr(sv))
    print(repr(tv))
    print(repr(uv))
    print(repr(vv))

    local = True 
    s_gs = FrameGensteps(s.genstep, s.sframe, local=local, symbol="s_gs" ) if not s is None else None
    t_gs = FrameGensteps(t.genstep, t.sframe, local=local, symbol="t_gs" ) if not t is None else None
    u_gs = FrameGensteps(u.genstep, u.sframe, local=local, symbol="u_gs" ) if not u is None else None
    v_gs = FrameGensteps(v.genstep, v.sframe, local=local, symbol="v_gs" ) if not v is None else None

    lim = FrameGensteps.CombineLim( [s_gs, t_gs, u_gs, v_gs] )

    s_frame = s.sframe if not s is None else None
    t_frame = t.sframe if not t is None else None
    u_frame = u.sframe if not u is None else None
    v_frame = v.sframe if not v is None else None

    frame = sframe.CombineFrame( [s_frame, t_frame, u_frame, v_frame ] )

    s_offset = efloatarray_("S_OFFSET", "0,0,0")
    t_offset = efloatarray_("T_OFFSET", "0,0,0")
    u_offset = efloatarray_("U_OFFSET", "0,0,0")
    v_offset = efloatarray_("V_OFFSET", "0,0,0")

    print("S_OFFSET: %s " % repr(s_offset))
    print("T_OFFSET: %s " % repr(t_offset))
    print("U_OFFSET: %s " % repr(u_offset))
    print("V_OFFSET: %s " % repr(v_offset))

    #aa = {}
    #aa[X] = efloatlist_("XX")
    #aa[Y] = efloatlist_("YY")
    #aa[Z] = efloatlist_("ZZ")

    s_hit = s.simtrace[:,0,3]>0 if not s is None else None
    t_hit = t.simtrace[:,0,3]>0 if not t is None else None
    u_hit = u.simtrace[:,0,3]>0 if not u is None else None
    v_hit = v.simtrace[:,0,3]>0 if not v is None else None

    s_pos = s_offset + s.simtrace[s_hit][:,1,:3] if not s is None else None
    t_pos = t_offset + t.simtrace[t_hit][:,1,:3] if not t is None else None
    u_pos = u_offset + u.simtrace[u_hit][:,1,:3] if not u is None else None
    v_pos = v_offset + v.simtrace[v_hit][:,1,:3] if not v is None else None

    topline = os.environ.get("TOPLINE", "cf_G4CXSimtraceTest.py")
    botline = os.environ.get("BOTLINE", "S_OFFSET:%s T_OFFSET:%s U_OFFSET:%s V_OFFSET:%s " % (str(s_offset),str(t_offset), str(u_offset), str(v_offset)))
    title = [topline, botline ]

    fig, ax = mp.subplots(figsize=SIZE/100.)
    fig.suptitle("\n".join(title))

    ax.set_aspect('equal')

    if not frame is None and not lim is None:
        H,V = frame.axes       # traditionally H,V = X,Z  but now generalized
        _H,_V = frame.axlabels

        ax.set_xlim( lim[H] )
        ax.set_ylim( lim[V] )
        ax.set_xlabel(_H)
        ax.set_ylabel(_V)
    pass

    if not s_pos is None:
        ax.scatter( s_pos[:,0], s_pos[:,2], label="S:%s" % s_geom, s=1 ) 
    pass
    if not t_pos is None:
        ax.scatter( t_pos[:,0], t_pos[:,2], label="T:%s" % t_geom, s=1 ) 
    pass
    if not u_pos is None:
        ax.scatter( u_pos[:,0], u_pos[:,2], label="U:%s" % u_geom, s=1 ) 
    pass
    if not v_pos is None:
        ax.scatter( v_pos[:,0], v_pos[:,2], label="V:%s" % v_geom, s=1 ) 
    pass


    ax.legend()
    fig.show()

