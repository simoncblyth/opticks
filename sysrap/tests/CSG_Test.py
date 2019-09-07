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
To regenerate the OpticksCSG.py::

    sysrap-;sysrap-cd 
    c_enums_to_python.py OpticksCSG.h  # check 
    c_enums_to_python.py OpticksCSG.h > OpticksCSG.py 


"""
from opticks.sysrap.OpticksCSG import CSG_


if __name__ == '__main__':

     for k, v in CSG_.raw_enum():
         vv = getattr(CSG_, k)
         print k, v, vv



     for i in range(20):
         d = CSG_.desc(i)
         i2 = CSG_.fromdesc(d)

         print "%3d %15s %d " % (i, d, i2)



