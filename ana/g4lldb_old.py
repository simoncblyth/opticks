#!/usr/bin/python
"""
## NB system python

g4lldb.py
=============

Caution with the PREFIXd comment strings in this
file, they are parsed by this file to extract the 
lldb startup command to define breakpoints.

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

from collections import defaultdict, OrderedDict

from opticks.ana.autobreakpoint import AutoBreakPoint
from opticks.ana.ucf import UCF
from opticks.ana.loc import Loc
from opticks.tools.lldb_ import lldb
from opticks.tools.evaluate import Evaluate, Value

log = logging.getLogger(__name__)


class Quote(object):
    PTN = re.compile("\"(\S*)\"")

    @classmethod
    def Extract(cls, s):
        q = cls(str(s))
        return q.q

    def __init__(self, s):
        match = self.PTN.search(str(s))
        self.q = match.group(1) if match else None
  


FMT = "// %80s : %s " 
COUNT = defaultdict(lambda:0)

REGISTER = {}
INSTANCE = defaultdict(lambda:[])



class Frm(object):
    """
    """
    @classmethod 
    def Split(cls, memkln):
        if memkln.find(":") > -1:
             mem, kln = memkln.split(":")
        else:
             mem, kln = memkln, None
        pass 
        return mem, kln

    def __init__(self, frame, pfx="", pframe=None):


        loc = Loc(pframe)

        self.frame = frame
        self.pfx = pfx
        self.pframe = pframe
        self.loc = loc
        self.this = frame.FindVariable("this")

    def __call__(self, keykln ):
        key, kln = self.Split(keykln)
        kls = REGISTER.get(kln, None) 
        if key[0] == ".":
            child = self.this.GetValueForExpressionPath(key)
        elif key[0] == "/":
            child = self.frame.FindVariable(key[1:])
        else:
            child = self.this.GetChildMemberWithName(key)
        pass
        mpfx = self.pfx+"->"+key
        obj = kls(child, mpfx, pframe=self.pframe) if kls is not None else child
        return obj


 
class QDict(OrderedDict):

    @classmethod
    def Dump(cls):
        for func, ls in INSTANCE.items():
            print FMT % ( func, len(ls) )
        pass

    
    loc = property(lambda self:self.frm.loc)
    tag = property(lambda self:self.frm.loc.tag)
    idx = property(lambda self:self.frm.loc.idx)

    def __init__(self,  this, pfx, pframe=None):
        """
        :param this: SBValue object 
        :param pfx: string prefix 
        :param pframe: python frame of calling breakpoint function sys._getframe()
        """
        OrderedDict.__init__(self)

        #print "QDict kls:%s pfx:%s " % (self.__class__.__name__, pfx )

        frame = this.GetFrame()
        loc = Loc(pframe)

        frm = Frm(frame, pfx, pframe)

        self.frm = frm 
        self.this = this
        self.pfx = pfx


        global INSTANCE
        if loc.func is not None:
            INSTANCE[loc.func].append(self)         
        pass

        if self.MEMBERS is None:
            return
        pass
        m = self.MEMBERS
        if "\n" in m:    ## multi-line raw members lists
            m = " ".join(m.replace("\n", " ").split())
        pass

        for keykln in m.split():
            key, kln = Frm.Split(keykln)

            #print "tag:%s key:%s kln:%s " % ( tag, key, kln ) 

            try:
                self[key] = str(frm(keykln))
            except TypeError:
                self[key] = "type-error" 
            pass
        pass

    def __repr__(self):
        lines = []
        if self.tag is not None:
            lines.append("")
            lines.append(self.loc.hdr)
        pass
        lines.append(FMT % (self.pfx, self.__class__.__name__))
        for k,v in self.items():
            brief = getattr(v, "BRIEF", False)
            composite = v.__class__ in REGISTER.values()
            if brief:
                lines.append(FMT % (k,v))
            elif composite:
                lines.append(FMT % (k, ""))
                lines.append("%s" % v)
            else:
                lines.append(FMT % (k,v))
            pass
        pass
        return "\n".join(lines)


class bool_(QDict):
    BRIEF = True
    MEMBERS = None
    def __repr__(self):
        s = self.this.GetValue()
        return str(s == "true")

class G4bool(bool_):
    pass

class int_(QDict):
    BRIEF = True
    MEMBERS = None
    def __repr__(self):
        return " %d " % int(self.this.GetValue())
 
class size_t(int_):
    pass

class G4int(int_):
    pass
 
class double_(QDict):
    BRIEF = True
    MEMBERS = None
    def __repr__(self):
        return " %g " % float(self.this.GetValue())  # GetValue yields str

class G4double(double_):
    pass



class G4ThreeVector(QDict):
    BRIEF = True
    MEMBERS = r"""
    dx:G4double
    dy:G4double
    dz:G4double
    """ 
    def __repr__(self):
        return  " (%s %s %s) " % ( self["dx"], self["dy"], self["dz"] )
        #return  " (%8.3f %8.3f %8.3f) " % ( float(self["dx"]), float(self["dy"]), float(self["dz"]) )
        






class string_(QDict):
    BRIEF = True
    MEMBERS = None
    def __repr__(self):
        return " %s " % Quote.Extract(self.this)
    
class G4String(string_):
    pass
      
class Enum(QDict):
    BRIEF = True
    MEMBERS = None
    def __repr__(self):
        return " %s " % self.this.GetValue()


class G4StepPoint(QDict):
    MEMBERS = r"""
    fPosition:G4ThreeVector 
    fGlobalTime:G4double 
    fMomentumDirection:G4ThreeVector 
    fPolarization:G4ThreeVector 
    fVelocity:G4double
    """

class G4Step(QDict):
    MEMBERS = r"""
    fpPreStepPoint:G4StepPoint 
    fpPostStepPoint:G4StepPoint
    """

class G4SteppingManager(QDict):
    MEMBERS = r"""
    fStep:G4Step 
    fStepStatus:Enum 
    PhysicalStep:G4double 
    fCurrentProcess:G4VProcess
    """

class G4TrackingManager(QDict):
    MEMBERS = r"""
    .fpSteppingManager.fCurrentProcess.theProcessName:G4String 

    .fpSteppingManager.fStep.fpPreStepPoint.fPosition:G4ThreeVector
    .fpSteppingManager.fStep.fpPreStepPoint.fGlobalTime:G4double

    .fpSteppingManager.fStep.fpPostStepPoint.fPosition:G4ThreeVector
    .fpSteppingManager.fStep.fpPostStepPoint.fGlobalTime:G4double
    """

    _MEMBERS = r"""
    fpSteppingManager:G4SteppingManager

    .fpSteppingManager.fStepStatus:Enum
    .fpSteppingManager.PhysicalStep:G4double
    .fpSteppingManager.fStep.fpPreStepPoint.fMomentumDirection:G4ThreeVector
    .fpSteppingManager.fStep.fpPostStepPoint.fMomentumDirection:G4ThreeVector
    """

class G4VProcess(QDict):
    MEMBERS = r"""
    theProcessName:G4String
    """




class CRandomEngine_cc_flatExit(QDict):
    MEMBERS = r"""
    .m_flat:double
    .m_curand_index:int
    .m_current_record_flat_count:int
    .m_current_step_flat_count:int
    .m_location:string
    """ 
    def __repr__(self):
        brief = " %3d %3d  : %s : %s " % ( 
                                  int(self[".m_current_record_flat_count"]),
                                  int(self[".m_current_step_flat_count"]),
                                  self[".m_flat"],
                                  self[".m_location"]
                                )
        return FMT % (self.tag, brief)




ENGINE = None
class CRandomEngine(object):
    def __init__(self, pindex):
        pass
        ucf = UCF( pindex )
        print repr(ucf)
        self.ucf = ucf 
        self.pindex = pindex

    def postTrack(self, func):
        finst = INSTANCE[func]
        for i, inst in enumerate(finst):
            print "%30s : %s " % ( i, inst )
            #print "%30s : %10s : %s " % ( i, inst[".m_curand_index"], inst )
        pass
 

def CRandomEngine_cc_flatExit_(frame, bp_loc, sess):
    """  
    flatExit
 
    (*lldb*) br set -f CRandomEngine.cc -l %(flatExit)s
    (*lldb*) br com add 1 -F opticks.ana.g4lldb.CRandomEngine_cc_flatExit_
    """

    e = Evaluate()
    v = Value(frame.FindVariable("this"))

    u_g4 = e(v("m_flat")) 
    loc_g4 = e(v("m_location")) 
    assert type(u_g4) is float 
    assert type(loc_g4) is str 

    crfc = e(v("m_current_record_flat_count")) 
    curi = e(v("m_curand_index")) 
    assert type(crfc) is int 
    assert type(curi) is int

    assert ENGINE is not None 
    lucf = len(ENGINE.ucf) 
    u = ENGINE.ucf[crfc-1] if crfc-1 < lucf else None 

    u_ok = u.fval if u is not None else -1
    loc_ok = u.lab  if u is not None else "ucf-overflow" 

    df = abs(u_g4 - u_ok) 
    misrng = df > 1e-6 
    misloc = loc_ok != loc_g4
    mrk = "%s%s" % ( "*" if misrng else "-", "#" if misloc else "-")

    print "flatExit: mrk:%2s crfc:%5d df:%.9g u_g4:%.9g u_ok:%.9g loc_g4:%20s loc_ok:%20s  : lucf : %d    " % ( mrk, crfc, df, u_g4, u_ok, loc_g4,loc_ok, lucf )
    print u 

    stop = mrk != "--"
    return stop



def CRandomEngine_cc_preTrack_(frame, bp_loc, sess):
    global ENGINE

    e = Evaluate()
    v = Value(frame.FindVariable("this"))
    curi = e(v("m_curand_index")) 
    assert type(curi) is int 
    print "curi:%d " % curi 

    ENGINE = CRandomEngine(pindex=curi)
    return False
    

def CRandomEngine_cc_postTrack_(frame, bp_loc, sess):
    ENGINE.postTrack("CRandomEngine_cc_flatExit_")
    return False





def _G4SteppingManager_cc_191_(frame, bp_loc, sess):
    """
    After DefinePhysicalStepLength() sets PhysicalStep and fStepStatus, before InvokeAlongStepDoItProcs()

    g4-;g4-cls G4SteppingManager 
    """
    stepMgr = G4SteppingManager(frame.FindVariable("this") , "this", sys._getframe() ) 
    print stepMgr
    return True


def _G4TrackingManager_cc_131_(frame, bp_loc, sess):
    """
    Step Conclusion : TrackingManager step loop just after Stepping() 

    g4-;g4-cls G4TrackingManager 
    """

    trackMgr = G4TrackingManager(frame.FindVariable("this") , "this", sys._getframe() ) 
    print trackMgr
    QDict.Dump() 

    return False




class OpRayleigh_cc_ExitPostStepDoIt(QDict):
    MEMBERS = r"""
    .theProcessName:G4String 
    .thePILfactor:G4double
    .aParticleChange.thePositionChange:G4ThreeVector
    .aParticleChange.thePolarizationChange:G4ThreeVector
    .aParticleChange.theMomentumDirectionChange:G4ThreeVector
   
    /rand:G4double
    /constant:G4double
    /cosTheta:G4double
    /CosTheta:G4double
    /SinTheta:G4double
    /CosPhi:G4double
    /SinPhi:G4double
    /OldMomentumDirection:G4ThreeVector
    /NewMomentumDirection:G4ThreeVector
    /OldPolarization:G4ThreeVector
    /NewPolarization:G4ThreeVector

    """

class OpRayleigh_cc_EndWhile(OpRayleigh_cc_ExitPostStepDoIt):
    pass


def _OpRayleigh_cc_EndWhile_(frame, bp_loc, sess):
    """
    EndWhile
    """
    inst = OpRayleigh_cc_EndWhile(frame.FindVariable("this") , "this", sys._getframe() )
    print inst
    return False


def _OpRayleigh_cc_ExitPostStepDoIt_(frame, bp_loc, sess):
    """
    ExitPostStepDoIt

    opticks-;opticks-cls OpRayleigh
    g4-;g4-cls G4VDiscreteProcess
    g4-;g4-cls G4VProcess
    """
    inst = OpRayleigh_cc_ExitPostStepDoIt(frame.FindVariable("this") , "this", sys._getframe() )
    print inst
    return False







class G4SteppingManager2_cc_181(QDict):
    MEMBERS = r"""
    .fCurrentProcess.theProcessName:G4String 
    .physIntLength:G4double 
    fCurrentProcess:G4VProcess 
    fCondition:Enum 
    fStepStatus:Enum 
    fPostStepDoItProcTriggered:size_t
    """
    _MEMBERS = "PhysicalStep"  # undef at this juncture

    @classmethod 
    def Dump(cls, pframe):

        func = pframe.f_code.co_name
        doc = pframe.f_code.co_consts[0]
        doclines = filter(None, doc.split("\n"))
        label = doclines[0].lstrip() if len(doclines) > 0 else "-"  # 1st line of docstring
     
        bpfunc = "%s_" % cls.__name__ 
        print "%s : %s " % ( bpfunc, label)
        
        qwn = ".fCurrentProcess.theProcessName .physIntLength"
        for i, inst in enumerate(INSTANCE[bpfunc]):        
            for q in qwn.split():
                print FMT % ( q, inst[q] )
            pass
        pass
        INSTANCE[bpfunc][:] = []    ## clear collected breakpoint states


def _G4SteppingManager2_cc_181_(frame, bp_loc, sess):
    """
    Collecting physIntLength within process loop after PostStepGPIL

    g4-;g4-cls G4SteppingManager 

    See:  notes/issues/stepping_process_review.rst 
    """
    inst = G4SteppingManager2_cc_181(frame.FindVariable("this"), "this", sys._getframe() )
    #print inst.tag
    #print inst 
    return False

def _G4SteppingManager2_cc_225_(frame, bp_loc, sess):
    """
    Dumping lengths collected by _181 after PostStep process loop 
    """
    G4SteppingManager2_cc_181.Dump(sys._getframe())
    return False


class G4SteppingManager2_cc_270(QDict):
    MEMBERS = r"""
    .fCurrentProcess.theProcessName:G4String 
    .physIntLength:G4double 
    .PhysicalStep:G4double 
    .fStepStatus:Enum
    """

def _G4SteppingManager2_cc_270_(frame, bp_loc, sess):
    """
    Near end of DefinePhysicalStepLength : Inside MAXofAlongStepLoops after AlongStepGPIL

    g4-;g4-cls G4SteppingManager
    """
    inst = G4SteppingManager2_cc_270(frame.FindVariable("this"), "this", sys._getframe() )
    print inst
    return False


class DsG4OpBoundaryProcess_cc_ExitPostStepDoIt(QDict):
    MEMBERS = r"""
    .OldMomentum:G4ThreeVector
    .NewMomentum:G4ThreeVector
    .theStatus
    """

def _DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_(frame, bp_loc, sess):
    """
    ExitPostStepDoIt

    opticks-;opticks-cls DsG4OpBoundaryProcess
    """
    inst = DsG4OpBoundaryProcess_cc_ExitPostStepDoIt(frame.FindVariable("this"), "this", sys._getframe() )
    print inst
    return False


class DsG4OpBoundaryProcess_cc_DiDiTransCoeff(QDict):
    MEMBERS = r"""
    .OldMomentum:G4ThreeVector
    .NewMomentum:G4ThreeVector
    /TransCoeff:G4double
    /_u:G4double
    /_transmit:bool_
    """

def DsG4OpBoundaryProcess_cc_DiDiTransCoeff_(frame, bp_loc, sess):
    """
    DiDiTransCoeff
    opticks-;opticks-cls DsG4OpBoundaryProcess
    """
    inst = DsG4OpBoundaryProcess_cc_DiDiTransCoeff(frame.FindVariable("this"), "this", sys._getframe() )
    print inst
    return False






class G4Navigator_ComputeStep_1181(QDict):
    MEMBERS = r"""
    .fNumberZeroSteps:G4int
    .fLastStepWasZero:G4bool
    """  

def _G4Navigator_ComputeStep_1181_(frame, bp_loc, sess):
    """
    ComputeStep return : NOT WORKING : MISSING THIS 
    """ 
    inst = G4Navigator_ComputeStep_1181(frame.FindVariable("this"), "this", sys._getframe() )
    print inst
    return True

    

class G4Transportation_cc_517(QDict):
    MEMBERS = r"""
    .fParticleChange.thePositionChange:G4ThreeVector
    """

    _MEMBERS = r"""
    /startPosition:G4ThreeVector
    /startMomentumDir:G4ThreeVector
    /newSafety:G4double
    .fGeometryLimitedStep:G4bool
    .fFirstStepInVolume:G4bool
    .fLastStepInVolume:G4bool
    .fMomentumChanged:G4bool
    .fShortStepOptimisation:G4bool
    .fTransportEndPosition:G4ThreeVector
    .fTransportEndMomentumDir:G4ThreeVector
    .fEndPointDistance:G4double
    .fParticleChange.thePositionChange:G4ThreeVector
    .fParticleChange.theMomentumDirectionChange:G4ThreeVector
    .fLinearNavigator.fNumberZeroSteps:G4int
    .fLinearNavigator.fLastStepWasZero:G4bool
    """


def _G4Transportation_cc_517_(frame, bp_loc, sess):
    """
    AlongStepGetPhysicalInteractionLength Exit 

    g4-;g4-cls G4Transportation
    """
    inst = G4Transportation_cc_517(frame.FindVariable("this"), "this", sys._getframe() )
    print inst
    return False




def G4VDiscreteProcess_PostStepGetPhysicalInteractionLength(frame, bp_loc, sess):
    """
    """
    name = "%s.%s " % ( __name__, sys._getframe().f_code.co_name  )
    proc = frame.FindVariable("this")
    procName = proc.GetChildMemberWithName("theProcessName")
    left = proc.GetChildMemberWithName("theNumberOfInteractionLengthLeft")
    print "%100s %s %s  " % ( name, procName, left )
    return False

def G4VProcess_ResetNumberOfInteractionLengthLeft(frame, bp_loc, sess):
    """
    """
    name = "%s.%s " % ( __name__, sys._getframe().f_code.co_name  )
    this = frame.FindVariable("this")
    procName = this.GetChildMemberWithName("theProcessName")

    print "%100s %s " % ( name, procName )
    return False


class Mock(object):
    def GetChildMemberWithName(self, name):
        return name

def test_G4StepPoint():
    dummy = Mock() 
    sp = G4StepPoint(dummy,"dummy")
    print sp  




REGISTER["G4ThreeVector"] = G4ThreeVector
REGISTER["double"] = double_
REGISTER["string"] = string_
REGISTER["int"] = int_
REGISTER["G4double"] = G4double
REGISTER["G4bool"] = G4bool
REGISTER["bool_"] = bool_
REGISTER["G4int"] = G4int
REGISTER["size_t"] = size_t
REGISTER["G4String"] = G4String
REGISTER["Enum"] = Enum
REGISTER["G4StepPoint"] = G4StepPoint
REGISTER["G4Step"] = G4Step
REGISTER["G4SteppingManager"] = G4SteppingManager
REGISTER["G4TrackingManager"] = G4TrackingManager
REGISTER["G4VProcess"] = G4VProcess
REGISTER["CRandomEngine"] = CRandomEngine






def test_Quote():
    """

    The exmaple docstring first line
    The exmaple docstring
    """

    s = "(G4String) theProcessName = (std::__1::string = \"OpBoundary\")"
    q = Quote.Extract(s)
    print q
    assert q == "OpBoundary"


def test_resolve_bp():
    print resolve_bp("CRandomEngine.cc", "flatExit")

if __name__ == '__main__':

    #logging.basicConfig(level=logging.INFO) 
    print AutoBreakPoint(path=__file__, module="opticks.ana.g4lldb")
    #test_Quote()
    #test_resolve_bp()











