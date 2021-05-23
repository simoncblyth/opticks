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

enu.py
=======

Extracts the strings and enumeration constants from headers 

::


    g4-
    enu.py --hdr $(g4-dir)/source/processes/optical/include/G4OpBoundaryProcess.hh  --kls CBoundaryProcess

    enu.py --hdr $(g4-dir)/source/processes/electromagnetic/utils/include/G4EmProcessSubType.hh --kls CEmProcessSubType



"""
import os, re, logging, sys, argparse
from collections import OrderedDict as odict 
log = logging.getLogger(__name__)



class KV(odict):
   def __init__(self, ls):
       odict.__init__(self)
       for item in ls:
           kv = list(map(str.strip, item.split("=")))
           assert len(kv) == 2 
           k, v = kv
           self[k] = v 
       pass


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
        self.kv = KV(self)

    def parse(self, path):
        lines = list(map(str.strip,open(path).readlines()))
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
                tokens = list(filter(None,map(str.strip, uline.split(","))))
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


    id = property(lambda self:"// enu.py --hdr %s " % self.path) 

    def labels(self, kls):
        return "\n".join([self.id]+ list(map(lambda _:self.labels_tmpl % dict(kls=kls,token=_,qtoken="\"%s\""%_, utoken="%s_"% _), self.kv.keys())) )

    def case(self, pfx):
        return "\n".join([self.id]+ list(map(lambda _:self.case_tmpl % dict(pfx=pfx,token=_, token_="%s_" % _), self.kv.keys())) )

    def scc(self):
        return "\n".join([self.id]+ list(map(lambda _:self.scc_tmpl % dict(token=_), self.kv.keys())) )



       



if __name__ == '__main__':

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument( "--hdr",  default=None )
    parser.add_argument( "--kls",  default="CBoundaryProcess" )
    parser.add_argument( "--ns",  default="Ds::" )
    parser.add_argument( "--level", default="info", help="logging level" ) 
    args = parser.parse_args()

    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)

    path = args.hdr

    l = Enu(path)
     
    print(l)
    print("\nlabels\n")
    print(l.labels(kls=args.kls))
    print("\ncase\n")
    print(l.case(pfx=args.ns))
    print("\ncase\n")
    print(l.case(pfx=""))
    print("\nscc\n")
    print(l.scc())

