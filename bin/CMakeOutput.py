#!/usr/bin/env python
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
::

    cat /tmp/thrap-expandTest-i | CMakeOutput.py 

"""
import sys, re, os, logging, argparse
log = logging.getLogger(__name__)



class Line(object):
    def __init__(self, txt, idx, secno=0):
        self.txt = txt
        self.idx = idx
        self.secno = secno
        self.is_hdr = len(txt) > 2 and txt[0:2] == "--"
        
        self.is_splay = txt.find("nvcc") > -1 or txt.find("c++") > -1
        self.ltxt = txt.replace(" "," \\\n")


    def __str__(self):
        return "[%3d;%d] %s " % ( self.idx, self.secno, self.ltxt if self.is_splay else self.txt) 
    def __repr__(self):
        return "[%3d;%d] %s " % ( self.idx, self.secno, self.txt) 


class Section(object):
    def __init__(self, title):
        self.title = title 
        self.lines = []
    def __repr__(self):
        return "(%s lines) " % len(self.lines) + repr(self.title)
    def __str__(self):
        return "\n".join(map(str,self.lines)) 



class CMakeOutput(object):
    def __init__(self, lines):
        """
        At section titles started by -- 
        collect any prior section and then start a new one
        """
        self.lines = lines
        self.sects = []
        sect = None
        for i, line in enumerate(lines):
            ln = Line(line, i+1, secno=len(self.sects))        
            if ln.is_hdr:
                if sect is not None:
                    self.sects.append(sect)
                pass
                sect = Section(ln)
                pass 
            else:
                if sect is not None:
                    sect.lines.append(ln)
                pass
            pass
        pass
        if sect is not None:
            self.sects.append(sect)
        pass

    

    def __str__(self):
        return "\n".join(map(str, self.sects)) 
    def __repr__(self):
        return "\n".join(map(repr, self.sects)) 



if __name__ == '__main__':

    lines = map(str.strip, sys.stdin.readlines() ) 
    #lines = map(str.strip, file(path,"r").readlines() ) 

    mo = CMakeOutput(lines)

    print repr(mo)
    #print "str----"
    #print str(mo)

    for isect, sect in enumerate(mo.sects):
        print "\n\n-------- %s : %s " % ( isect, repr(sect) ) 
        print str(sect)


 

