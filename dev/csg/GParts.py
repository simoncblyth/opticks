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

TODO: consolidate ana/prim.py with dev/csg/GParts.py 


"""
import os
import numpy as np

class GParts(object):
   def __init__(self, base):
       self.name = base
       self.partBuf = np.load(os.path.expandvars(os.path.join(base,"partBuffer.npy")))
       self.primBuf = np.load(os.path.expandvars(os.path.join(base,"primBuffer.npy")))
   pass
   def __getitem__(self, i):
       if i >= 0 and i < len(self.primBuf):
           prim = self.primBuf[i]
           partOffset, numPart, a, b = prim
           return self.partBuf[partOffset:partOffset+numPart]
       else:
           raise IndexError
       pass 


if __name__ == '__main__':

   assert 0, "superceded by ana/GParts.py ana/prim.py, THIS dev/csg/GParts IS TO BE DELETED" 
   p = GParts("/tmp/blyth/opticks")
   p0 = p[0]
   p1 = p[1]





