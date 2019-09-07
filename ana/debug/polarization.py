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
"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_environment
from opticks.ana.evt import Evt

deg = np.pi/180.


if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)
    opticks_environment()

    spol, ppol = "5", "6"
    g = Evt(tag="-"+spol, det="rainbow", label="S G4")
    o = Evt(tag=spol, det="rainbow", label="S Op")

    # check magnitude of polarization
    for e in [g,o]: 
        mag = np.linalg.norm(e.rpol_(0),2,1)
        assert mag.max() < 1.01 and mag.min() > 0.99



