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


import os
from collections import OrderedDict as odict 


class Matcher(object):
    def __init__(self):
        exts = "py c cc cpp hpp hh h glsl cu m mm sh bash cmake mac png txt rst old dead pyc log in err out".split()
        self.exts = exts    
        d = odict()
        for ext in exts: 
            d[ext] = 0   
        pass
        d["OTHER"] = 0  
        d["TOTAL"] = 0  
        self.d = d 
        self.other = []

    def __call__(self, path):
        ext = os.path.splitext(path)[1]
        ext = ext[1:] # remove "." 

        if ext in self.d:
            self.d[ext] += 1 
        else:
            self.d["OTHER"] += 1
            self.other.append(path) 
        pass
        self.d["TOTAL"] += 1 
    
    def traverse(self, base):
        for root, dirs, names in os.walk(base):
            if root.find(".hg") > -1:continue
            for name in names:
                path = os.path.join(root, name)
                #print(path)
                self(path)   
            pass
        pass 
        self.check()

    def check(self):
        total = 0 
        for k, v in self.d.items():
            if k != "TOTAL":
                total += v        
            pass
        pass
        assert total == self.d["TOTAL"] 

    def __repr__(self):
        return "\n".join([" %10s : %d " % ( kv[0], kv[1]) for kv in sorted(self.d.items(), key=lambda kv:kv[1], reverse=True)]) 

if __name__ == '__main__':

    m = Matcher()
    m.traverse(os.path.expanduser("~/opticks"))

    #print("\n".join(m.other))
    print(m)

