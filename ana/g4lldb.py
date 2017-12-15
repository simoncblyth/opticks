#!/usr/bin/python
"""
## NB system python

g4lldb.py
=============


TODO : adopt point-by-point interleaved logging
---------------------------------------------------

* split by step in ucf.py and interleave at that level, 
  as its too difficult to grok at per-rng level



Using lldb breakpoint python scripting
---------------------------------------

::

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 -DD

        # special -DD debug launching: 
        #
        # 1. update the breakpoint setup, via parsing this file
        # 2. source the commands on starting lldb 


Automated breakpoint scripting
---------------------------------

The below is done on every op.sh launch with -D option
and when envvar OPTICKS_LLDB_SOURCE is defined.

::

    delta:~ blyth$ g4lldb.py 
    # generated from-and-by /Users/blyth/opticks/ana/g4lldb.py 
    command script import opticks.ana.g4lldb
    br set -f CRandomEngine.cc -l 210
    br com add 1 -F opticks.ana.g4lldb.CRandomEngine_cc_210
    br set -f G4TrackingManager.cc -l 131
    br com add 2 -F opticks.ana.g4lldb.G4TrackingManager_cc_131


Background on g4lldb python scripting
-----------------------------------

* moved to env-/lldb-vi as got too long 
* see also env-/tools/lldb_/standalone.py for development of evaluation functions


    >>> from opticks.tools.evaluate import Evaluate
    >>> ev = Evaluate()
    >>> ev.evaluate_frame(lldb.frame)
    >>> this = lldb.frame.FindVariable("this")
    >>> ec = ev.evaluate_comp(this)


"""

import os, sys, logging, re

from opticks.ana.autobreakpoint import AutoBreakPoint
from opticks.ana.ucf import UCF
from opticks.tools.lldb_ import lldb
from opticks.tools.evaluate import Evaluate, Value, EV
from opticks.tools.loc import Loc

log = logging.getLogger(__name__)

ENGINE = None
def CRandomEngine_cc_preTrack_(frame, bp_loc, sess):
    """
    preTrack label
    """
    global ENGINE
    ENGINE = CRandomEngine()
    ploc = Loc(sys._getframe(), __name__)
    return ENGINE.preTrack(ploc, frame, bp_loc, sess)

def CRandomEngine_cc_flat_(frame, bp_loc, sess):
    """
    flat label
    """
    global ENGINE
    ploc = Loc(sys._getframe(), __name__)
    return ENGINE.flat(ploc,frame, bp_loc, sess)
    
def CRandomEngine_cc_jump_(frame, bp_loc, sess):
    """
    jump label
    """
    global ENGINE
    ploc = Loc(sys._getframe(), __name__)
    return ENGINE.jump(ploc,frame, bp_loc, sess)
 
def CRandomEngine_cc_postTrack_(frame, bp_loc, sess):
    """
    postTrack label
    """
    global ENGINE
    ploc = Loc(sys._getframe(), __name__)
    return ENGINE.postTrack(ploc,frame, bp_loc, sess)


class CRandomEngine(EV):
    def __init__(self):
        EV.__init__(self, None)

    def preTrack(self, ploc, frame, bp_loc, sess):
        self.v = frame.FindVariable("this")
        pindex = self.ev("m_curand_index") 
        assert type(pindex) is int 
        print "pindex:%d " % pindex
        ucf = UCF( pindex )
        lucf = len(ucf) 
        print repr(ucf)
        self.ucf = ucf 
        self.lucf = lucf
        self.pindex = pindex
        print "%s lucf:%d pindex:%d" % (ploc.tag, lucf, pindex ) 
        return False

    def postTrack(self, ploc, frame, bp_loc, sess):
        print ploc.hdr
        self.v = frame.FindVariable("this")
        pindex = self.ev("m_curand_index") 
        assert pindex == self.pindex
        print "%s pindex:%d" % (ploc.tag, pindex ) 
        return False

    def jump(self, ploc, frame, bp_loc, sess):
        self.v = frame.FindVariable("this")

        cursor_old = self.ev("m_cursor_old") 
        offset = self.ev("m_offset") 
        offset_count = self.ev("m_offset_count") 
        cursor = self.ev("m_cursor") 

        print "%s cursor_old:%d offset:%d offset_count:%d cursor:%d " % (ploc.tag, cursor_old, offset, offset_count, cursor ) 
        return False

    def flat(self, ploc, frame, bp_loc, sess):
        self.v = frame.FindVariable("this")

        u_g4 = self.ev("m_flat") 
        loc_g4 = self.ev("m_location") 
        assert type(u_g4) is float 
        assert type(loc_g4) is str 

        crf = self.ev("m_current_record_flat_count") 
        csf = self.ev("m_current_step_flat_count") 
        cur = self.ev("m_curand_index") 

        assert type(crf) is int 
        assert type(csf) is int 
        assert type(cur) is int and cur == self.pindex

        u = self.ucf[crf] if crf < self.lucf else None 

        u_ok = u.fval if u is not None else -1
        loc_ok = u.lab  if u is not None else "ucf-overflow" 

        df = abs(u_g4 - u_ok) 
        misrng = df > 1e-6 
        misloc = loc_ok != loc_g4
        mrk = "%s%s" % ( "*" if misrng else "-", "#" if misloc else "-")

        f_ = lambda f:'{0:.9f}'.format(f)
        g_ = lambda g:'{0:.10g}'.format(g)

        print "%s mrk:%2s crf:%2d csf:%2d loc_g4/ok: ( %30s %30s ) df:%18s u_g4/ok:( %s %s ) " % ( ploc.tag, mrk, crf, csf, loc_g4, loc_ok, g_(df), f_(u_g4), f_(u_ok)   )
        #print u 

        stop = mrk != "--"
        return stop



if __name__ == '__main__':
    print AutoBreakPoint(path=__file__, module="opticks.ana.g4lldb")











