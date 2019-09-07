#!/usr/bin/python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

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

