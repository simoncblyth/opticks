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


import numpy as np

def test_repeat():
    # see tests/NPYTest.cc:test_repeat

    aa = np.load("/tmp/aa.npy")
    bb = np.load("/tmp/bb.npy")
    cc = np.load("/tmp/cc.npy")

    bbx = np.repeat(aa, 10, axis=0) ;
    assert np.all(bb == bbx)

    ccx = np.repeat(aa, 10, axis=0).reshape(-1,10,1,4)
    assert np.all(cc == ccx)



if __name__ == '__main__':
    test_repeat()



