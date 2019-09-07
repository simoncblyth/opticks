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


TORCH = 4096 

import os, numpy as np
from opticks.ana.base import opticks_main

if __name__ == '__main__':
    args = opticks_main()
 
    a = np.load(os.path.expandvars("$TMP/trivialCheck.npy"))
    i = a.view(np.int32)

    ## these are specific to default TORCH genstep

    assert np.all(i[:,2,0] == TORCH)
    assert np.all(i[:,2,3] == 100000)   

    assert np.all( np.arange(len(i), dtype=np.int32) == i[:,3,0] )    # photon_id
    assert np.all( np.arange(len(i), dtype=np.int32)*4 == i[:,3,1] )  # photon_offset

    assert np.all(i[:,3,2] == 0)           # genstep_id  (all zero as only 1 genstep for default TORCH)
    assert np.all(i[:,3,3] == i[:,3,2]*6)  # genstep_offset




