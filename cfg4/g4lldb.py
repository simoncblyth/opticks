#!/usr/bin/env python
"""
See
    (lldb) help br com add

Add the handler::

    (lldb) command script import opticks.cfg4.g4lldb
             ## put into ~/.lldbinit to avoid having to repeat this

    (lldb) b G4VDiscreteProcess::PostStepGetPhysicalInteractionLength 
             ## create pending breakpoint

    (lldb) br com  add 1 -F opticks.cfg4.g4lldb.brk    
             ## add command to pending breakpoint 

"""

import sys
from collections import defaultdict

FMT = "// %50s : %s " 
COUNT = defaultdict(lambda:0)

def py_G4SteppingManager_DefinePhysicalStepLength(frame, bp_loc, sess):
    """

    See:  notes/issues/stepping_process_review.rst 

    ::

        g4-;g4-cls G4SteppingManager 
        g4-;g4-cls G4SteppingManager2
 
        tboolean-;tboolean-box --okg4 --align -D


        (lldb) b -f G4SteppingManager2.cc -l 181

            ## inside process loop after PostStepGPIL call giving physIntLength and fCondition

        (lldb) br com  add 1 -F opticks.cfg4.g4lldb.py_G4SteppingManager_DefinePhysicalStepLength 


    Seems can access member vars, but so far not general stack items, other than "this" ?
    """
    name = sys._getframe().f_code.co_name 

    global COUNT 
    COUNT[name] += 1 

    kvar = "physIntLength fCondition PhysicalStep fStepStatus fPostStepDoItProcTriggered"

    this = frame.FindVariable("this")
    proc = this.GetChildMemberWithName("fCurrentProcess")
    procName = proc.GetChildMemberWithName("theProcessName")

    print "//" 
    print FMT % ( name, COUNT[name] )
    print FMT % ( "procName", procName ) 

    for k in kvar.split():
        #v = frame.FindVariable(k)    gives no-value
        v = this.GetChildMemberWithName(k)
        print FMT % ( k, v )
    pass
    return False




def py_G4VDiscreteProcess_PostStepGetPhysicalInteractionLength(frame, bp_loc, sess):
    """
    ::

        b G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
        br com  add 1 -F opticks.cfg4.g4lldb.py_G4VDiscreteProcess_PostStepGetPhysicalInteractionLength

    """
    name = "py_G4VDiscreteProcess_PostStepGetPhysicalInteractionLength"
    proc = frame.FindVariable("this")
    procName = proc.GetChildMemberWithName("theProcessName")
    left = proc.GetChildMemberWithName("theNumberOfInteractionLengthLeft")
    print "%100s %s %s  " % ( name, procName, left )
    return False


def py_G4VProcess_ResetNumberOfInteractionLengthLeft(frame, bp_loc, sess):
    """

        b G4VProcess::ResetNumberOfInteractionLengthLeft
        br com  add 1 -F opticks.cfg4.g4lldb.py_G4VProcess_ResetNumberOfInteractionLengthLeft    
 
    """
    name = "py_G4VProcess_ResetNumberOfInteractionLengthLeft"
    this = frame.FindVariable("this")
    procName = this.GetChildMemberWithName("theProcessName")
    print "%100s %s " % ( name, procName )
    return False



 

