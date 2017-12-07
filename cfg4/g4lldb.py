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





    G4VProcess_ResetNumberOfInteractionLengthLeft (G4String) theProcessName = (std::__1::string = "Scintillation")
    G4VProcess_ResetNumberOfInteractionLengthLeft (G4String) theProcessName = (std::__1::string = "OpBoundary")
    G4VProcess_ResetNumberOfInteractionLengthLeft (G4String) theProcessName = (std::__1::string = "OpRayleigh")
    G4VProcess_ResetNumberOfInteractionLengthLeft (G4String) theProcessName = (std::__1::string = "OpAbsorption")

    G4VProcess_ResetNumberOfInteractionLengthLeft (G4String) theProcessName = (std::__1::string = "Scintillation")
    G4VProcess_ResetNumberOfInteractionLengthLeft (G4String) theProcessName = (std::__1::string = "OpBoundary")

    G4VProcess_ResetNumberOfInteractionLengthLeft (G4String) theProcessName = (std::__1::string = "Scintillation")
    G4VProcess_ResetNumberOfInteractionLengthLeft (G4String) theProcessName = (std::__1::string = "OpBoundary")


"""




def py_G4VDiscreteProcess_PostStepGetPhysicalInteractionLength(frame, bp_loc, sess):
    """
    ::

        b G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
        br com  add 1 -F opticks.cfg4.g4lldb.py_G4VDiscreteProcess_PostStepGetPhysicalInteractionLength

    """
    name = "py_G4VDiscreteProcess_PostStepGetPhysicalInteractionLength"
    this = frame.FindVariable("this")
    procName = this.GetChildMemberWithName("theProcessName")
    left = this.GetChildMemberWithName("theNumberOfInteractionLengthLeft")
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



 

