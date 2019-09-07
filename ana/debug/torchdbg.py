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



Torchstep are read by optixrap/cu/torchstep.h:tsload


::

    In [2]: run torchdbg.py

    In [3]: a
    Out[3]: 
    array([[[   0. ,    0. ,    0. ,    0. ],
            [   0. ,    0. , -200. ,    0.1],             ## post : ts.x0, ts.t0
            [   0. ,    0. ,    1. ,    1. ],             ## dirw : ts.p0, ts.weight
            [   0. ,    0. ,   -1. ,  550. ],             ## polw : ts.pol, ts.wavelength
            [   0.5,    1. ,    0. ,    1. ],             ## zeaz : ts.zeaz
            [ 100. ,   25. ,    0. ,    0. ]]],           ## beam : ts.beam.x (radius) ts.beam.y (distance)  ts.type eg T_REFLTEST ts.mode
             dtype=float32)

    In [4]: a.view(np.int32)
    Out[4]: 
    array([[[       4096,           0,           3,       10000],     ## ctrl :    GenstepType/ts.ParentId/ts.MaterialIndex/ts.NumPhotons
            [          0,           0, -1018691584,  1036831949],
            [          0,           0,  1065353216,  1065353216],
            [          0,           0, -1082130432,  1141473280],
            [ 1056964608,  1065353216,           0,  1065353216],
            [ 1120403456,  1103626240,           5,           9]]]                 ts.mode = 5 , ts.type = 9 
         , dtype=int32)

::

    In [8]: 0x1 << 2 | 0x1 << 0   ## M_SPOL | M_FLAT_THETA
    Out[8]: 5




::

      3 // for both non-CUDA and CUDA compilation
      4 typedef enum {
      5    T_UNDEF,                       # 0
      6    T_SPHERE,
      7    T_POINT,
      8    T_DISC,
      9    T_DISC_INTERSECT_SPHERE,       # 4
     10    T_DISC_INTERSECT_SPHERE_DUMB,  # 5
     11    T_DISCLIN,
     12    T_DISCAXIAL,
     13    T_INVSPHERE,
     14    T_REFLTEST,                    # 9 
     15    T_INVCYLINDER,
     16    T_RING, 
     17    T_NUM_TYPE
     18 }               Torch_t ;

     20 typedef enum {
     21    M_UNDEF         = 0x0 ,
     22    M_SPOL          = 0x1 << 0,
     23    M_PPOL          = 0x1 << 1,
     24    M_FLAT_THETA    = 0x1 << 2,
     25    M_FLAT_COSTHETA = 0x1 << 3,
     26    M_FIXPOL        = 0x1 << 4
     27 }              Mode_t ;
     28     






::

simon:optickscore blyth$ cat OpticksPhoton.h 

    enum
    {
        CERENKOV          = 0x1 <<  0,    
        SCINTILLATION     = 0x1 <<  1,    
        ...
        TORCH             = 0x1 << 12,    ## 4096
        G4GUN             = 0x1 << 14
    }; 





"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

if __name__ == '__main__':
     os.environ["TMP"] = os.path.expandvars("/tmp/$USER/opticks")
     a = np.load(os.path.expandvars("$TMP/torchdbg.npy"))


