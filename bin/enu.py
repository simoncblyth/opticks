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

    g4- ; ./enu.py $(g4-dir)/source/processes/optical/include/G4OpBoundaryProcess.hh 


"""
import os, re, logging, sys
log = logging.getLogger(__name__)

class Enu(list):
    #begin = "{" 
    begin = "enum" 
    end = "};"
    def __init__(self, path):
        list.__init__(self)

        self.path = path 
        path = os.path.expandvars(path)

        self.state = -1
        self.last = False
        self.parse(path)

    def parse(self, path):
        lines = map(str.strip,file(path).readlines())
        for line in lines:

            uline = line
            if line.startswith(self.begin):
                uline = line[len(self.begin):] 
                self.state = 0
                pos = uline.find("{") 
                if pos > -1:
                    uline = uline[pos+1:]
                else:
                    uline = ""
                pass
            elif line.endswith(self.end):
                self.state = -1
                self.last = True
                uline = line[:-len(self.end)]
            elif self.state > -1: 
                self.state += 1 
            else:
                pass
            pass          

            uline = uline.replace("{","")

            taketoken = ( self.state >= 0 or self.last ) and len(uline) > 0 
            self.last = False
            tokens = []
            if taketoken:
                tokens = filter(None,map(str.strip, uline.split(",")))
                #print "%3d : %30s : [%s] " % ( self.state, uline, repr(tokens)  )
            pass
            if len(tokens)>0:
                self.extend(tokens)
            pass

    def __str__(self):
        return "\n".join([self.id]+self) 


    labels_tmpl = str.strip(r""" 
    const char* %(kls)s::%(utoken)-35s = %(qtoken)-35s ;
    """)

    case_tmpl = str.strip(r"""
    case %(pfx)s%(token)-35s : s = %(token_)-35s ; break ;
    """) 

    scc_tmpl = str.strip(r"""
    static const char* %(token)s_ ; 
    """)


    id = property(lambda self:"// enu.py %s " % self.path) 

    def labels(self, kls):
        return "\n".join([self.id]+ map(lambda _:self.labels_tmpl % dict(kls=kls,token=_,qtoken="\"%s\""%_, utoken="%s_"% _), self) )

    def case(self, pfx):
        return "\n".join([self.id]+ map(lambda _:self.case_tmpl % dict(pfx=pfx,token=_, token_="%s_" % _), self) )

    def scc(self):
        return "\n".join([self.id]+ map(lambda _:self.scc_tmpl % dict(token=_), self) )





if __name__ == '__main__':
     logging.basicConfig(level=logging.INFO)  

     path = sys.argv[1] if len(sys.argv) > 1 else "DsG4OpBoundaryProcessStatus.h"

     l = Enu(path)
     
     print l
     print
     print l.labels(kls="CBoundaryProcess")
     print
     print l.case(pfx="Ds::")
     print
     print l.case(pfx="")
     print
     print l.scc()


    

