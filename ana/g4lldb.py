#!/usr/bin/python
"""
## NB system python

g4lldb.py
=============


THOUGHTS ON CODE FOR DEBUGGING
------------------------------

* resist adding code just for debugging access, instead find where 
  the state you want to get at is already 


Using lldb breakpoint python scripting
---------------------------------------

::

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 -DD

        # special -DD debug launching: 
        #
        # 1. update the breakpoint setup, via parsing this file
        # 2. source the commands on starting lldb 


Background on g4lldb python scripting
-----------------------------------

* moved to env-/lldb-vi as got too long 
* see also env-/tools/lldb_/standalone.py for development of evaluation functions


::

    >>> from opticks.tools.evaluate import EV ; ev = EV(lldb.frame.FindVariable("this"))


"""

import os, sys, logging, re, inspect

from opticks.ana.ucf import UCF
from opticks.ana.bouncelog import BounceLog

from opticks.tools.lldb_ import lldb
from opticks.tools.autobreakpoint import AutoBreakPoint
from opticks.tools.evaluate import Evaluate, Value, EV
from opticks.tools.loc import Loc

log = logging.getLogger(__name__)

ENGINE = None
def CRandomEngine_cc_preTrack(frame, bp_loc, sess):
    """
    preTrack label
    """
    global ENGINE
    ENGINE = CRandomEngine()
    ploc = Loc(sys._getframe(), __name__)
    return ENGINE.preTrack(ploc, frame, bp_loc, sess)

def CRandomEngine_cc_flat(frame, bp_loc, sess):
    """
    flat label
    """
    ploc = Loc(sys._getframe(), __name__)
    return ENGINE.flat(ploc,frame, bp_loc, sess)
    
def CRandomEngine_cc_jump(frame, bp_loc, sess):
    """
    jump label
    """
    ploc = Loc(sys._getframe(), __name__)
    return ENGINE.jump(ploc,frame, bp_loc, sess)

def CRandomEngine_cc_postStep(frame, bp_loc, sess):
    """
    postStep label
    """
    ploc = Loc(sys._getframe(), __name__)
    return ENGINE.postStep(ploc,frame, bp_loc, sess)
 
def CRandomEngine_cc_postTrack(frame, bp_loc, sess):
    """
    postTrack label
    """
    ploc = Loc(sys._getframe(), __name__)
    return ENGINE.postTrack(ploc,frame, bp_loc, sess)





def _G4SteppingManager_cc_191(frame, bp_loc, sess):
    """
    After DefinePhysicalStepLength() sets PhysicalStep and fStepStatus, before InvokeAlongStepDoItProcs()

    g4-;g4-cls G4SteppingManager 
    """
    ploc = Loc(sys._getframe(), __name__)

    self = EV(frame.FindVariable("this"))
    fStepStatus = self.ev("fStepStatus")

    print "%s : %20s : %s " % (ploc.tag, fStepStatus, ploc.label)
    return False

def _DsG4OpBoundaryProcess_cc_StepTooSmall(frame, bp_loc, sess):
    ploc = Loc(sys._getframe(), __name__)
    self = EV(frame.FindVariable("this"))
    theStatus = self.ev("theStatus")

    print "%s : %20s : %s " % (ploc.tag, theStatus, ploc.label)
    return True


def _CSteppingAction_cc_setStep(frame, bp_loc, sess):
    """ 
    """
    ploc = Loc(sys._getframe(), __name__)
    print "%s :  %s " % (ploc.tag, ploc.label)
    self = EV(frame.FindVariable("this"))
    return False


def _CRec_cc_add(frame, bp_loc, sess):
    """ 
    
    """
    ploc = Loc(sys._getframe(), __name__)

    self = EV(frame.FindVariable("this"))
    bst = self.ev("m_boundary_status")
    pri = self.ev("m_prior_boundary_status")
    print "%s : bst:%20s pri:%20s : %s " % (ploc.tag, bst, pri, ploc.label)
    return False




class CRandomEngine(EV):
    def __init__(self):
        EV.__init__(self, None)

    def preTrack(self, ploc, frame, bp_loc, sess):
        self.v = frame.FindVariable("this")
        pindex = self.ev("m_curand_index") 
        assert type(pindex) is int 

        ucf = UCF( pindex )
        lucf = len(ucf) 
        
        self.bounce_log = BounceLog( pindex )
         
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
        jump_ = self.ev("m_jump") 
        jump_count = self.ev("m_jump_count") 
        cursor = self.ev("m_cursor") 

        print "%s cursor_old:%d jump_:%d jump_count:%d cursor:%d " % (ploc.tag, cursor_old, jump_, jump_count, cursor ) 
        return False

    def postStep(self, ploc, frame, bp_loc, sess):
        self.v = frame.FindVariable("this")

        sid = self.ev(".m_ctx._step_id")
        pri = self.ev(".m_ctx._prior_boundary_status")
        bst = self.ev(".m_ctx._boundary_status")
        opt = self.ev("m_okevt_pt")

        print "%s step_id:%d bst:%15s pri:%15s okevt_pt:%s " % (ploc.tag, sid, bst, pri, opt ) 
        bounce = self.bounce_log.get(sid, None) 
        dump = False
        if bounce is not None and dump:
            print bounce
        pass
        print "--"  # postStep spacer 
        return False

    def flat(self, ploc, frame, bp_loc, sess):
        self.v = frame.FindVariable("this")

        ug4 = self.ev("m_flat") 
        lg4 = self.ev("m_location") 
        cur = self.ev("m_cursor") 
        crf = self.ev("m_current_record_flat_count") 
        csf = self.ev("m_current_step_flat_count") 
        cix = self.ev("m_curand_index") 

        idx = cur  
        # correspondence between sequences 
        #    crf: gets offset by the kludge
        #    cur: is always the real flat cursor index

        assert type(crf) is int 
        assert type(csf) is int 
        assert type(cix) is int and cix == self.pindex

        u = self.ucf[idx] if idx < self.lucf else None 
        uok = u.fval if u is not None else -1
        lok = u.lab  if u is not None else "ucf-overflow" 

        df = abs(ug4 - uok) 
        misrng = df > 1e-6 
        misloc = lok != lg4
        mrk = "%s%s" % ( "*" if misrng else "-", "#" if misloc else "-")

        f_ = lambda f:'{0:.9f}'.format(f)
        g_ = lambda g:'{0:.10g}'.format(g)

        print "%s cix:%5d mrk:%2s cur:%2d crf:%2d csf:%2d lg4/ok: ( %30s %30s ) df:%18s ug4/ok:( %s %s ) " % ( ploc.tag, cix, mrk, cur, crf, csf, lg4, lok, g_(df), f_(ug4), f_(uok)   )
        #print u 

        stop = mrk != "--"
        #return stop
        return False



if __name__ == '__main__':
    print AutoBreakPoint(path=__file__, module="opticks.ana.g4lldb")











