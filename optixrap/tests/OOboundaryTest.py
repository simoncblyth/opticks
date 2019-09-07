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

   ipython -i $(optixrap-sdir)/tests/OOboundaryTest.py 

"""

import os, numpy as np
from opticks.ana.base import opticks_main


def load(tag, name, reshape=None):
    dir_ = os.path.expandvars("$TMP/%s" % name)
    path = os.path.join(dir_,"%s.npy" % tag)
    a = np.load(path)
    log.info(" load %s %s %s " % (tag, repr(a.shape), path))
    if reshape is not None:
        a = a.reshape(reshape)
        log.info("reshape %s to %s " % (tag, repr(a.shape)))
    pass
    return a 

if __name__ == '__main__':
    args = opticks_main()
    #name = "OOboundaryTest" 
    name = "OOboundaryLookupTest" 

    ori = load("ori", name)
    inp = load("inp", name)
    out = load("out", name, reshape=ori.shape)

    if not np.allclose(ori, out):
        for i in range(len(ori)):
            ok = np.allclose(out[i],ori[i])
            print "%4d %s \n" % (i, ok)
        pass
    pass
    assert np.allclose(ori, out)







