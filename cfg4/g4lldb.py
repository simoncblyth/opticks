#!/usr/bin/env python
"""
See
    (lldb) help br com add

Add the handler::

    (lldb) command script import opticks.cfg4.g4lldb
             ## put into ~/.lldbinit to avoid having to do this

    (lldb) b G4VDiscreteProcess::PostStepGetPhysicalInteractionLength 
             ## create pending breakpoint

    (lldb) br com  add 1 -F opticks.cfg4.g4lldb.brk    
             ## add command to pending breakpoint 

"""

def brk(frame, bp_loc, sess):
    print "frame", frame
    this = frame.FindVariable("this")
    procName = this.GetChildMemberWithName("theProcessName")
    print procName
    stop = False
    return stop


