#!/usr/bin/env python
"""
cfg4lldb.py
=============

Caution with the PREFIXd comment strings in this
file, they are parsed by this file to extract the 
lldb startup command to define breakpoints.


TODO
--------

* simplify handling of atomic values, perhaps with V wrapper


Using lldb breakpoint python scripting
---------------------------------------

::

    unset OPTICKS_LLDB_SOURCE
    export OPTICKS_LLDB_SOURCE=/tmp/g4lldb.txt  

        # eg put these lines into ~/.bash_profile
        # toggle commenting the export to switch on/off

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 -D

        # normal debug launching, this will: 
        #
        # 1. update the breakpoint setup, via parsing this file
        # 2. source the commands on starting lldb 



Automated breakpoint scripting
---------------------------------

The below is done on every op.sh launch with -D option
and when envvar OPTICKS_LLDB_SOURCE is defined.

::

    delta:~ blyth$ cfg4lldb.py 
    # generated from-and-by /Users/blyth/opticks/ana/cfg4lldb.py 
    command script import opticks.ana.cfg4lldb
    br set -f CRandomEngine.cc -l 210
    br com add 1 -F opticks.ana.cfg4lldb.CRandomEngine_cc_210
    br set -f G4TrackingManager.cc -l 131
    br com add 2 -F opticks.ana.cfg4lldb.G4TrackingManager_cc_131



Experience with lldb python scripting
----------------------------------------

Not all variables available, use "frame variables" to find some, 
forturnately "this" has so far always been available.

::


    (lldb) frame variable

    (G4int) moduloFactor = <no location, value may have been optimized out>

    (G4Transportation *) this = 0x0000000110226ae0

    (const G4Track &) track = <variable not available>

    (G4double)  = <invalid load address>

    (G4double) currentMinimumStep = <variable not available>




Background on lldb python scripting
-----------------------------------

    (lldb) help br com add

    (lldb) script

    >>> help(lldb.frame)
    >>> help(lldb.SBValue)  # eg this

    (lldb) script
    Python Interactive Interpreter. To exit, type 'quit()', 'exit()' or Ctrl-D.
    >>> print lldb.frame
    frame #0: 0x000000010528fd8b libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength(this=0x0000000111a550b0) + 299 at G4SteppingManager2.cc:181

    >>> this = lldb.frame.FindVariable("this")

    >>> print "\n".join([str(this.GetChildAtIndex(_)) for _ in range(this.GetNumChildren())])

    (G4bool) KillVerbose = true
    (G4UserSteppingAction *) fUserSteppingAction = 0x000000010d369370
    (G4VSteppingVerbose *) fVerbose = 0x0000000111a55440
    (G4double) PhysicalStep = 1.7976931348623157E+308
    ...

    >>> memb_ = lambda this:"\n".join([str(this.GetChildAtIndex(_)) for _ in range(len(this))])

    >>> print memb_(this)

    (G4bool) KillVerbose = true
    (G4UserSteppingAction *) fUserSteppingAction = 0x000000010d369370
    (G4VSteppingVerbose *) fVerbose = 0x0000000111a55440
    ...

    >>> p = this.GetChildMemberWithName("fCurrentProcess")
    >>> print p
    (G4VProcess *) fCurrentProcess = 0x000000010d3ad880

    >>> re.compile("\"(\S*)\"").search(str(this.GetValueForExpressionPath(".fCurrentProcess.theProcessName"))).group(1)
    'OpBoundary'


NOW AUTOMATED : Connecting the handlers
--------------------------------------------


Add the handler::

    (*lldb*) command script import opticks.ana.cfg4lldb
        ## formerly put into ~/.lldbinit 

    (lldb) b G4VDiscreteProcess::PostStepGetPhysicalInteractionLength 
        ## create pending breakpoint

    (lldb) br com  add 1 -F opticks.ana.cfg4lldb
        ## add command to pending breakpoint 

"""

import os, sys, logging, re, fnmatch
from collections import defaultdict, OrderedDict

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


class Parse(dict):
    PREFIX = "    (*lldb*)"
    BR_SET = "br set"
    BR_COM_ADD_1_ = "br com add 1 "
    PTN = re.compile("-f (\S*) -l \%\((\S*)\)")

    @classmethod
    def resolve_path(cls, patn):
        base = os.environ["OPTICKS_HOME"]
        paths = []
        for path, dirs, files in os.walk(os.path.abspath(base)):
            for filename in fnmatch.filter(files, patn):
                paths.append(os.path.join(path, filename))
            pass
        pass
        assert len(paths) == 1
        return paths[0]

    @classmethod
    def resolve_bpline( cls, fnptn, mkr ):
        """
        Finds the source and looks for the marker, to give the lineno
        of the breakpoint.  Remember to recompile after changing source!
        """
        path = cls.resolve_path(fnptn)
        marker = "// (*lldb*) %s" % mkr
        #print "marker:[%s]" % marker
        nls = filter( lambda nl:nl[1].find(marker) > -1, enumerate(file(path).readlines()) )
        l1 = int(nls[0][0]) + 1 if len(nls) == 1 else None
        return l1 

    @classmethod
    def resolve_mkr(cls, line):
        """
        br set -f CRandomEngine.cc -l %(flatExit)s
        """
        match = cls.PTN.search(line)
        assert match, "line fails to match : [%s] " % line 
        fnm = match.group(1)
        mkr = match.group(2)
        lno = cls.resolve_bpline(fnm, mkr)
        return fnm,mkr,lno    
                    

    def __init__(self):
        dict.__init__(self)
        self["N"] = 0 
        lines = filter(lambda l:l.startswith(self.PREFIX), file(__file__).readlines())
        lines = map(lambda l:l[len(self.PREFIX):].rstrip().lstrip(), lines)
        self.lines = []
        for line in lines:
            is_br_set = line.startswith(self.BR_SET)
            is_br_com_add = line.startswith(self.BR_COM_ADD_1_)
            #print "# is_br_set:%d is_br_com_add:%d  : %s " % ( is_br_set, is_br_com_add, line )
            if is_br_set:
                self["N"] += 1 
                if line.find("%(") > -1:
                    fnm,mkr,lno = self.resolve_mkr(line)
                    self[mkr] = lno
                pass
            if is_br_com_add:
                line = line.replace(" 1 ", " %(N)s ")
            pass
            self.lines.append( line % self )
        pass
    def __repr__(self):
        hdr = "# generated from-and-by %s " % ( os.path.abspath(__file__) )  
        return "\n".join([hdr]+self.lines)





     

 
class QDict(OrderedDict):

    @classmethod
    def Tag(cls, func):
        if func is None:return None
        pass
        name = "%s.%s" % ( __name__, func  )
        global COUNT 
        tag = "%s.[%d]" % ( name, COUNT[name] )
        COUNT[name] += 1 
        return tag


    @classmethod
    def Hdr(cls, tag, label=""):
        return FMT % (tag, label)

    @classmethod
    def Dump(cls):
        for func, ls in INSTANCE.items():
            print FMT % ( func, len(ls) )
        pass


    def __init__(self,  this, pfx, pframe=None):
        """
        :param this: SBValue object 
        :param pfx: string prefix 
        :param pframe: python frame of calling breakpoint function sys._getframe()
        """
        OrderedDict.__init__(self)



        if pframe is not None:
            func = pframe.f_code.co_name
            doc = pframe.f_code.co_consts[0]
            doclines = filter(None, doc.split("\n"))
            label = doclines[0].lstrip() if len(doclines) > 0 else "-"  # 1st line of docstring
        else:
            func = None
            label = "-"
        pass

        tag = self.Tag(func) 

        frame = this.GetFrame()

        self.this = this
        self.pfx = pfx
        self.tag = tag
        self.label = label

        global INSTANCE
        if func is not None:
            INSTANCE[func].append(self)         
        pass

        if self.MEMBERS is None:
            return
        pass
        m = self.MEMBERS
        ## allow multi-line raw members lists
        if "\n" in m:    
            m = " ".join(m.replace("\n", " ").split())
        pass

        for memkln in m.split():

            if memkln.find(":") > -1:
                 mem, kln = memkln.split(":")
            else:
                 mem, kln = memkln, None
            pass 

            kls = REGISTER.get(kln, None) 

            mpfx = pfx+"->"+mem

            capture = False
            if mem[0] == ".":
                child = this.GetValueForExpressionPath(mem)
                capture = True
            elif mem[0] == "/":
                child = frame.FindVariable(mem[1:])
            else:
                child = this.GetChildMemberWithName(mem)
            pass
            self[mem] = kls(child, mpfx) if kls is not None else child
            if capture:  ## dont rely on underlying obj existance, capture immediately 
                self[mem] = str(self[mem]) 
            pass
        pass

    def __repr__(self):
        lines = []
        if self.tag is not None:
            lines.append("")
            lines.append(self.Hdr(self.tag, self.label))
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


class G4ThreeVector(QDict):
    BRIEF = True
    #MEMBERS = "dx:G4double dy:G4double dz:G4double" 
    MEMBERS = "dx dy dz" 
    def __repr__(self):
        return  " (%8.3f %8.3f %8.3f) " % ( float(self["dx"].GetValue()), float(self["dy"].GetValue()), float(self["dz"].GetValue()) )
        

class size_t(QDict):
    BRIEF = True
    MEMBERS = None
    def __repr__(self):
        return " %d " % int(self.this.GetValue())


class G4int(QDict):
    BRIEF = True
    MEMBERS = None
    def __repr__(self):
        return " %d " % int(self.this.GetValue())
 
 
class G4bool(QDict):
    BRIEF = True
    MEMBERS = None
    def __repr__(self):
        s = self.this.GetValue()
        return str(s == "true")


class G4double(QDict):
    BRIEF = True
    MEMBERS = None
    def __repr__(self):
        return " %g " % float(self.this.GetValue())  # GetValue yields str
    
class G4String(QDict):
    BRIEF = True
    MEMBERS = None
    def __repr__(self):
        return " %s " % Quote.Extract(self.this)
       
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
    .fpSteppingManager.fStepStatus:Enum
    .fpSteppingManager.PhysicalStep:G4double
    .fpSteppingManager.fCurrentProcess.theProcessName:G4String 

    .fpSteppingManager.fStep.fpPreStepPoint.fPosition:G4ThreeVector
    .fpSteppingManager.fStep.fpPreStepPoint.fGlobalTime:G4double
    .fpSteppingManager.fStep.fpPreStepPoint.fMomentumDirection:G4ThreeVector

    .fpSteppingManager.fStep.fpPostStepPoint.fPosition:G4ThreeVector
    .fpSteppingManager.fStep.fpPostStepPoint.fGlobalTime:G4double
    .fpSteppingManager.fStep.fpPostStepPoint.fMomentumDirection:G4ThreeVector
    """

    _MEMBERS = r"""
    fpSteppingManager:G4SteppingManager
    """

class G4VProcess(QDict):
    MEMBERS = r"""
    theProcessName:G4String
    """

class CRandomEngine(QDict):
    MEMBERS = r"""
    m_flat 
    m_current_record_flat_count
    m_current_step_flat_count
    """ 

    def __repr__(self):
        brief = " %3d %3d  : %s " % ( 
                                  int(self["m_current_record_flat_count"].GetValue()),
                                  int(self["m_current_step_flat_count"].GetValue()),
                                  self["m_flat"].GetValue()
                                )
        return FMT % (self.tag, brief)


REGISTER["G4ThreeVector"] = G4ThreeVector
REGISTER["G4double"] = G4double
REGISTER["G4bool"] = G4bool
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



def CRandomEngine_cc_flatExit_(frame, bp_loc, sess):
    """   
    (*lldb*) br set -f CRandomEngine.cc -l %(flatExit)s
    (*lldb*) br com add 1 -F opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_
    """
    engine = CRandomEngine(frame.FindVariable("this") , "this", sys._getframe() ) 
    print engine
    return False

def G4SteppingManager_cc_191_(frame, bp_loc, sess):
    """
    After DefinePhysicalStepLength() sets PhysicalStep and fStepStatus, before InvokeAlongStepDoItProcs()

    (lldb) br set -f G4SteppingManager.cc -l 191
    (lldb) br com add 1 -F opticks.ana.cfg4lldb.G4SteppingManager_cc_191_

    g4-;g4-cls G4SteppingManager 
    """
    stepMgr = G4SteppingManager(frame.FindVariable("this") , "this", sys._getframe() ) 
    print stepMgr
    return True


def G4TrackingManager_cc_131_(frame, bp_loc, sess):
    """
    Step Conclusion : TrackingManager step loop just after Stepping() 

    (*lldb*) br set -f G4TrackingManager.cc -l 131
    (*lldb*) br com add 1 -F opticks.ana.cfg4lldb.G4TrackingManager_cc_131_

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


def OpRayleigh_cc_EndWhile_(frame, bp_loc, sess):
    """
    EndWhile

    (*lldb*) br set -f OpRayleigh.cc -l %(EndWhile)s
    (*lldb*) br com add 1 -F opticks.ana.cfg4lldb.OpRayleigh_cc_EndWhile_

    """
    inst = OpRayleigh_cc_EndWhile(frame.FindVariable("this") , "this", sys._getframe() )
    print inst
    return False


def OpRayleigh_cc_ExitPostStepDoIt_(frame, bp_loc, sess):
    """
    ExitPostStepDoIt

    (*lldb*) br set -f OpRayleigh.cc -l %(ExitPostStepDoIt)s
    (*lldb*) br com add 1 -F opticks.ana.cfg4lldb.OpRayleigh_cc_ExitPostStepDoIt_

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


def G4SteppingManager2_cc_181_(frame, bp_loc, sess):
    """
    Collecting physIntLength within process loop after PostStepGPIL

    (*lldb*) br set -f G4SteppingManager2.cc -l 181
    (*lldb*) br com add 1 -F opticks.ana.cfg4lldb.G4SteppingManager2_cc_181_

    g4-;g4-cls G4SteppingManager 

    See:  notes/issues/stepping_process_review.rst 
    """
    inst = G4SteppingManager2_cc_181(frame.FindVariable("this"), "this", sys._getframe() )
    #print inst.tag
    #print inst 
    return False

def G4SteppingManager2_cc_225_(frame, bp_loc, sess):
    """
    Dumping lengths collected by _181 after PostStep process loop 

    (*lldb*) br set -f G4SteppingManager2.cc -l 225
    (*lldb*) br com add 1 -F opticks.ana.cfg4lldb.G4SteppingManager2_cc_225_
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

def G4SteppingManager2_cc_270_(frame, bp_loc, sess):
    """
    Near end of DefinePhysicalStepLength : Inside MAXofAlongStepLoops after AlongStepGPIL

    (*lldb*) br set -f G4SteppingManager2.cc -l 270
    (*lldb*) br com add 1 -F opticks.ana.cfg4lldb.G4SteppingManager2_cc_270_

    g4-;g4-cls G4SteppingManager
    """
    inst = G4SteppingManager2_cc_270(frame.FindVariable("this"), "this", sys._getframe() )
    print inst
    return False


class DsG4OpBoundaryProcess_cc_736(QDict):
    MEMBERS = r"""
    .OldMomentum:G4ThreeVector
    .NewMomentum:G4ThreeVector
    """

def DsG4OpBoundaryProcess_cc_736_(frame, bp_loc, sess):
    """
    OpBoundary.PostStepDoIt return 

    (*lldb*) br set -f DsG4OpBoundaryProcess.cc -l 736
    (*lldb*) br com add 1 -F opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_736_

    opticks-;opticks-cls DsG4OpBoundaryProcess
     
    """
    inst = DsG4OpBoundaryProcess_cc_736(frame.FindVariable("this"), "this", sys._getframe() )
    print inst
    return False



class G4Navigator_ComputeStep_1181(QDict):
    MEMBERS = r"""
    .fNumberZeroSteps:G4int
    .fLastStepWasZero:G4bool
    """  

def G4Navigator_ComputeStep_1181_(frame, bp_loc, sess):
    """
    ComputeStep return : NOT WORKING : MISSING THIS 

    (lldb) br set -f G4Navigator.cc -l 1181
    (lldb) br com add 1 -F opticks.ana.cfg4lldb.G4Navigator_ComputeStep_1181_

    """ 
    inst = G4Navigator_ComputeStep_1181(frame.FindVariable("this"), "this", sys._getframe() )
    print inst
    return True

    

class G4Transportation_cc_517(QDict):
    MEMBERS = r"""
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

def G4Transportation_cc_517_(frame, bp_loc, sess):
    """
    AlongStepGetPhysicalInteractionLength Exit 

    (*lldb*) br set -f G4Transportation.cc -l 517
    (*lldb*) br com add 1 -F opticks.ana.cfg4lldb.G4Transportation_cc_517_

    g4-;g4-cls G4Transportation
 

    Find available variables, members with::

        (lldb) frame variable

        (lldb) script
        >>> this = lldb.frame.FindVariable("this")
        >>> memb_ = lambda this:"\n".join([str(this.GetChildAtIndex(_)) for _ in range(len(this))])
        >>> print memb_(this)

    """
    inst = G4Transportation_cc_517(frame.FindVariable("this"), "this", sys._getframe() )
    print inst
    return False




def dev_G4SteppingManager2_cc_181(frame, bp_loc, sess):
    tag = QDict.Tag( sys._getframe().f_code.co_name  )
    print QDict.Hdr(tag)

    this = frame.FindVariable("this")
    proc = this.GetChildMemberWithName("fCurrentProcess")
    procName = Quote.Extract(proc.GetChildMemberWithName("theProcessName"))
    print FMT % ( "procName", procName ) 
    
    for k in kvar.split():
        v = this.GetChildMemberWithName(k)
        print FMT % ( k, v )
    pass
    return True


def G4VDiscreteProcess_PostStepGetPhysicalInteractionLength(frame, bp_loc, sess):
    """
    (lldb) br set G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
    (lldb) br com add 1 -F opticks.ana.cfg4lldb.G4VDiscreteProcess_PostStepGetPhysicalInteractionLength
    """
    name = "%s.%s " % ( __name__, sys._getframe().f_code.co_name  )
    proc = frame.FindVariable("this")
    procName = proc.GetChildMemberWithName("theProcessName")
    left = proc.GetChildMemberWithName("theNumberOfInteractionLengthLeft")
    print "%100s %s %s  " % ( name, procName, left )
    return False

def G4VProcess_ResetNumberOfInteractionLengthLeft(frame, bp_loc, sess):
    """
    (lldb) br set G4VProcess::ResetNumberOfInteractionLengthLeft
    (lldb) br com add 1 -F opticks.ana.cfg4lldb.G4VProcess_ResetNumberOfInteractionLengthLeft    
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


def Tail(frame, bp_loc, sess):
    """   
    (lldb) r
    ##  does launch, BUT not connected to cli tty : so no output/input and leaves zombie process when interrupt
    """
    pass



def test_Introspect(pframe):
    func = pframe.f_code.co_name
    doc = pframe.f_code.co_consts[0]

    doclines = filter(None, doc.split("\n"))
    label = doclines[0].lstrip() if len(doclines) > 0 else "-"

    print "doc:[%s]" % doc
    print "func:[%s]" % func
    print "label:[%s]" % label

 
def test_Quote():
    """

    The exmaple docstring first line
    The exmaple docstring
    """
    test_Introspect(sys._getframe())

    s = "(G4String) theProcessName = (std::__1::string = \"OpBoundary\")"
    q = Quote.Extract(s)
    print q
    assert q == "OpBoundary"


def test_resolve_bp():
    print resolve_bp("CRandomEngine.cc", "flatExit")

if __name__ == '__main__':
    print Parse()
    #test_Quote()
    #test_resolve_bp()











