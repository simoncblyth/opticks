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

class G4PhysicsVector ; 
template <typename T> class GProperty ; 

/**
X4Property
===================

Converts ggeo.GProperty<T> into G4PhysicsVector

1. input ggeo.GProperty<T> uses wavelength domain in nanometers 
   in wavelength ascending order (which is reversed wrt energy)

2. output G4PhysicsVector using standard Geant4 energy units (MeV)
   with energies in ascending order.

**/

template <typename T>
class X4_API X4Property
{
    public:
        static G4PhysicsVector*  Convert(const GProperty<T>* prop) ; 
};

