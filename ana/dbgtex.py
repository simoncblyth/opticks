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

    export OPTICKS_KEYDIR=/usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/c250d41454fba7cb19f3b83815b132c2/1
    ipython -i -- $(which dbgtex.py) 

"""
import os, numpy as np

if __name__ == '__main__':

    t = np.load(os.path.expandvars("$OPTICKS_KEYDIR/dbgtex/buf.npy"))
    print t 
    o = np.load(os.path.expandvars("$OPTICKS_KEYDIR/dbgtex/obuf.npy"))
    print o




