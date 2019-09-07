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
bouncelog.py
==============

Parse the kernel print log::

     tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 --pindexlog -DD   

          ## write kernel pindexlog for photon 1230

    boucelog.py 1230

          ## parse the log 


"""
from __future__ import print_function
from collections import OrderedDict
import os, sys, re


class Bounce(list):
    def __init__(self):
        list.__init__(self)
    def __str__(self):
       return "\n".join([""]+self+[""])


class BounceLog(OrderedDict):
    @classmethod
    def printlogpath(cls, pindex):
        return os.path.expandvars("$TMP/ox_%s.log" % pindex )

    BOUNCE = re.compile("bounce:(\S*)")

    def __init__(self, pindex):
        OrderedDict.__init__(self)
        self.pindex = pindex
        self.path = self.printlogpath(pindex)
        self.parse(self.path)
        
    def parse(self, path):
        self.lines = map(lambda line:line.rstrip(),file(path).readlines())

        curr = []
        bounce = -1

        for i, line in enumerate(self.lines):
            m = self.BOUNCE.search(line)
            if m:
                #bounce = int(m.group(1))   ## some OptiX rtPrintf bug makes bounce always 0 
                bounce += 1
                self[bounce] = Bounce()
            pass
            if bounce > -1:
                self[bounce].append(line)
            pass
            #print(" %3d : %3d : %s " % ( i, bounce,  line ))


if __name__ == '__main__':


    pindex = int(sys.argv[1]) if len(sys.argv) > 1 else 1230

    bl = BounceLog(pindex)

    for k, v in bl.items():
        print(k)
        print(v)
        print("\n\n") 

   

