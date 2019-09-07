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
#include <vector>
#include <string>

class G4VSolid ; 

/**
X4SolidList
=============

Unlike most X4 classes there is no directly corresponding G4 class
which is converted from. G4SolidStore is somewhat related.

X4SolidList is used from the X4PhysicalVolume TraverseVolumeTree
structure traverse to collect G4VSolid instances.

**/

class X4_API X4SolidList
{
    public:
        X4SolidList(); 
        void addSolid(G4VSolid* solid); 
        bool hasSolid(G4VSolid* solid) const ;
        std::string desc() const ; 
        unsigned getNumSolids() const ; 
    private:
        std::vector<G4VSolid*> m_solidlist ; 
};

