/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

#include "G4RotationMatrix.hh"
#include "X4_API_EXPORT.hh"

// subclass to use protected ctor

template <typename T>
struct X4_API X4RotationMatrix  : public G4RotationMatrix 
{
     X4RotationMatrix( 
           T xx, T xy, T xz,
           T yx, T yy, T yz, 
           T zx, T zy, T zz 
     )
     : 
       G4RotationMatrix( xx, xy, xz ,   
                         yx, yy, yz ,
                         zx, zy, zz )
    {}

};

