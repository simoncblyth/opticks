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

from opticks.ana.xmatlib import XMatLib
from opticks.ana.PropLib import PropLib

if __name__ == '__main__':


    xma = XMatLib("/tmp/test.dae")

    mat = "MineralOil"

    g = xma[mat]["GROUPVEL"]

    n = xma[mat]["RINDEX"]




    
if 0:
    mlib = PropLib("GMaterialLib")

    mat = mlib("MineralOil")

    n = mat[:,0]
  
    w = mlib.domain

    dn = n[1:] - n[0:-1]

    dw = w[1:] - w[0:-1]



