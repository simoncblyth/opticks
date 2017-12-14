#!/usr/bin/python
# OSX system python, for lldb 

import os, sys
DIR = "/Library/Developer/CommandLineTools/Library/PrivateFrameworks/LLDB.framework/Resources/Python"
if os.path.isdir(DIR):
    sys.path.append(DIR)
pass

from collections import OrderedDict
import lldb
error = lldb.SBError()

from opticks.tools.standalone import Code, Standalone
from opticks.tools.evaluate import evaluate_frame, evaluate_var, evaluate_obj

if __name__ == '__main__':

    nam = "G4ThreeVectorTest"

    dir_ = os.path.dirname(__file__)
    src = os.path.abspath(os.path.join(dir_, "%s.cc" % nam))
    exe = os.path.expandvars("$OPTICKS_INSTALL_PREFIX/lib/%s" % nam)

    co = Code(src=src, exe=exe)
    print co

    st = Standalone(co, dump=True)  

    o = st.frame.FindVariable("o")
    print o 

    eo = evaluate_obj(o, vdump=True, error=error)
    print eo  


    #ef = evaluate_frame(st.frame, vdump=True, error=error)
    #print ef  

