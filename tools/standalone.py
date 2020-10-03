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

"""
NB thats the system python, not the macports one

* https://lldb.llvm.org/python-reference.html

::

    /usr/bin/python -i standalone.py 

    >>> print thread.GetFrameAtIndex(2)
    frame #2: 0x0000000107d59ff3 standalone`main(argc=1, argv=0x00007fff57ea6f38) + 83 at standalone.cc:28

"""

import os, sys
from collections import OrderedDict

from opticks.tools.lldb_ import lldb
from opticks.tools.evaluate import Evaluate



class Code(dict):
    @classmethod
    def find_bpline(cls, path, mkr ):
        marker = "// (*lldb*) %s" % mkr

        lines = file(path).readlines()
        #print "\n".join(lines)
        nls = filter( lambda nl:nl[1].find(marker) > -1, enumerate(lines) )
        assert len(nls) == 1
        return int(nls[0][0])+1 

    def compile_(self):
        print("compiling : %(cmd)s " % self)
        rc = os.system(self["cmd"])
        assert rc == 0, rc

    mkr = property(lambda self:self["mkr"])
    nam = property(lambda self:self["nam"])
    exe = property(lambda self:self["exe"])
    src = property(lambda self:self["src"])
    bpl = property(lambda self:self["bpl"])

    def __repr__(self):
        return r"""

             nam : %(nam)s
             mkr : %(mkr)s
             exe : %(exe)s
             src : %(src)s
             cmd : %(cmd)s
             bpl : %(bpl)s


        """ % self 


    def __init__(self, *args, **kwa):
        """
        When using an existing executable, 
        require *exe* path to the binary and *src* path to the source.

        When compiling standalone code 
        provide only the *src*  path.
        """
        dict.__init__(self, *args, **kwa)

        assert hasattr(self, "src")
        assert os.path.isfile(self["src"])
        self["nam"] = os.path.splitext(os.path.basename(self["src"]))[0]

        if not hasattr(self,"exe"):
            self["exe"] = "/tmp/%(nam)s" % self     
            self["cmd"] = "cc %(src)s -g -lc++ -o %(exe)s " % self # without -g fails to find breakpoints
            self.compile_()
        else:
            self["cmd"] = "-"
        pass
        if not hasattr(self, "mkr"):
            self["mkr"] = "Exit"
        pass
        self["bpl"] = self.find_bpline(self.src, self.mkr)

        



class Standalone(object):
    def __init__(self, code, dump=False):

        exe = code.exe
        bpl = code.bpl
        src = code.src

        debugger = lldb.SBDebugger.Create()
        debugger.SetAsync (False)

        target = debugger.CreateTargetWithFileAndArch (exe, lldb.LLDB_ARCH_DEFAULT)
        if dump:
            print("target:%s" % target)
        assert target

        filename = target.GetExecutable().GetFilename()
        if dump:
            print("filename:%s" % filename)

        bp = target.BreakpointCreateByLocation(src, bpl )    # needs name.cc filename lacks .cc
        if dump:
            print(bp)

        process = target.LaunchSimple (None, None, os.getcwd())   # synchronous mode returns at bp 
        if dump:
            print("process:%s" % process)
        assert process

        state = process.GetState ()
        if dump:
            print("state:%s" % state)
        assert state == lldb.eStateStopped

        thread = process.GetThreadAtIndex (0)
        if dump:
            print("thread:%s" % thread)
        assert thread

        frame = thread.GetFrameAtIndex (0)
        if dump:
            print("frame:%s" % frame)
        assert frame

        function = frame.GetFunction()
        if dump:
            print("function:%s" % function)
        assert function
            

        self.code = code
        self.debugger = debugger
        self.target = target
        self.process = process
        self.thread = thread
        self.frame = frame
        self.function = function


if __name__ == '__main__':

    #co = Code(src="standalone.cc")  ## when compiling 
    co = Code(src="standalone.cc", exe="/tmp/standalone")  ## when using preexistinge exe
    print(co)

    st = Standalone(co)  
    error = lldb.SBError()

    ev = Evaluate(error, opt="f")
    ef = ev.evaluate_frame(st.frame)

    #print ef 
    #print ef["o"]["_s"]

    ##  /usr/bin/python -i standalone.py   OR lldb-i 
   
    #v = st.frame.FindVariable("cc")
    




