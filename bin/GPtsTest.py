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

   ipython $(which GPtsTest.py) -i -- 0


"""
import os, sys 
import numpy as np

if __name__ == '__main__':

    imm = sys.argv[1] if len(sys.argv) > 0 else 0
    path = os.path.expandvars("$TMP/GGeo/GPtsTest/%(imm)s" % locals())
    name = "idxBuffer.npy"
    print(path)

    apath = os.path.join(path, "parts", name)
    bpath = os.path.join(path, "parts2", name)

    a = np.load(apath)
    b = np.load(bpath)

    au = len(np.unique(a[:,0]))
    bu = len(np.unique(b[:,0]))
   
    print("A:%s %s au:%d" % (a.shape,apath,au) )
    print("B:%s %s bu:%d" % (b.shape,bpath,bu) )
     
    print(a[:10])
    print(b[:10])

    assert np.all( a[:,1] == b[:,1])
    assert np.all( a[:,2] == b[:,2])
    assert np.all( a[:,3] == b[:,3])


