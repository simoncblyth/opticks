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


import os, numpy as np
np.set_printoptions(linewidth=200)
#np.set_printoptions(suppress=True, precision=3)
from opticks.ana.base import opticks_main

if __name__ == '__main__':
    args = opticks_main()
    bb = np.load("$TMP/OBndLib_convert_bndbuf.npy")
    print bb.shape
    print bb[13,0]


