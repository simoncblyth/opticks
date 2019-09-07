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

#/usr/bin/env python

import numpy as np

# see tests/GColorsTest.cc

if __name__ == '__main__':

   a = np.load("/tmp/colors_GBuffer.npy") 
   b = np.load("/tmp/colors_NPY.npy")

   assert a.shape == (256, 1)
   assert b.shape == (256, 4)

   aa = a.view(np.uint8)

   assert aa.shape == (256, 4)

   assert np.all( b == aa ) 
 

   print "a[:10]", a[:10]
   print "b", b
   print "aa", aa


