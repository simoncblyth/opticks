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

see also  dev/csg/postorder.py

::

   8 : 00001000 
   9 : 00001001 
   4 : 00000100 
  10 : 00001010 
  11 : 00001011 
   5 : 00000101 
   2 : 00000010 
  12 : 00001100 
  13 : 00001101 
   6 : 00000110 
  14 : 00001110 
  15 : 00001111 
   7 : 00000111 
   3 : 00000011 
   1 : 00000001 

"""
from opticks.bin.ffs import clz_

height = 3

depth_ = lambda i:31 - clz_(i)    ## count leading zeros 

elevation_ = lambda i:height - depth_(i)

next_ = lambda i:i >> 1 if i & 1 else (i << elevation_(i)) + (1 << elevation_(i))

leftmost_ = lambda h:1 << h 

bin_ = lambda _:"{0:08b}".format(_)


if __name__ == '__main__':
   
    i = leftmost_(height)
    while i > 0:
        print "%4s : %s " % ( i, bin_(i) )
        i = next_(i)
    pass


