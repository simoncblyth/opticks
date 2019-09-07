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
See also ana/geocache.bash 
"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

idp_ = lambda _:os.path.expandvars("$IDPATH/%s" % _ )
idp2_ = lambda _:os.path.expandvars("$IDPATH2/%s" % _ )



def cflib(aa, bb):
    """
    Compare buffers between two geocache
    """
    assert aa.shape == bb.shape
    print aa.shape

    for i in range(len(aa)):
        a = aa[i]  
        b = bb[i]  
        assert len(a) == 2 
        assert len(b) == 2 

        g0 = a[0] - b[0] 
        g1 = a[1] - b[1] 

        assert g0.shape == g1.shape

        print i, g0.shape, "g0max: ", np.max(g0), "g1max: ", np.max(g1)
    pass


def cflib_test():
    #rel = "GMaterialLib/GMaterialLib.npy"
    rel = "GSurfaceLib/GSurfaceLib.npy"

    aa = np.load(idp_(rel))
    bb = np.load(idp2_(rel))
    cflib(aa,bb)
 


if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)

    assert 0, "ancient code depending in IDPATH"

