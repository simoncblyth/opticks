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

#include "X4_API_EXPORT.hh"
#include "plog/Severity.h"

class G4OpticalSurface ; 
class GOpticalSurface ; 

/**
X4OpticalSurface
==================

CAUTION : Only a small fraction of Geant4 optical surface handling 
has been ported to Opticks.

**/

class X4_API X4OpticalSurface 
{
        static const plog::Severity LEVEL ; 
    public:
        // static const char* Type(G4SurfaceType type);  moved to X4SurfaceProperty::Name
        static GOpticalSurface* Convert(const G4OpticalSurface* const src );
};



