#!/usr/bin/env python
"""

https://lldb.llvm.org/python-reference.html

http://www.fabianguerra.com/ios/introduction-to-lldb-python-scripting/

::

    (lldb) command script import opticks.cfg4.print_command
    The "print_frame" python command has been installed and is ready for use.
    (lldb) 

    (lldb) command script import ~/opticks/cfg4/print_command.py
    The "print_frame" python command has been installed and is ready for use.
    (lldb) 


    (lldb) b G4VDiscreteProcess::PostStepGetPhysicalInteractionLength
    Breakpoint 1: no locations (pending).
    WARNING:  Unable to resolve breakpoint to any actual locations.
    (lldb) br command add 1
    Enter your debugger command(s).  Type 'DONE' to end.
    > print_frame 
    > DONE
    (lldb) r

::

    (lldb) br command add 1
    Enter your debugger command(s).  Type 'DONE' to end.
    > print_frame 
    > DONE
    (lldb) c

    (lldb) br command add 1 -s python -o "print_frame"



root = lldb.frame.FindVariable("this")
print "root", root
procName = root.GetChildMemberWithName("theProcessName")
print "procName", procName

    :param frame: lldb.SBFrame
    :param bp_loc: lldb.SBBreakpointLocation 
    :param sess: dict


"""


def __lldb_init_module(debugger, internal_dict):  
    debugger.HandleCommand('command script add -f print_command.print_frame print_frame')  
    print 'The "print_frame" python command has been installed and is ready for use.'  

def print_frame(debugger, command, result, internal_dict):
    print "print_frame"


def print_frame_1(debugger, command, result, internal_dict):

    print >>result, "print_frame"
    target = debugger.GetSelectedTarget()
    process = target.GetProcess()
    thread = process.GetSelectedThread()

    for frame in thread:
        print >>result, str(frame)
    pass
 




