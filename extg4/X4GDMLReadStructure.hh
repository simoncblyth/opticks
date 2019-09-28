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
#include <string>

class G4VSolid ; 

/**
X4GDMLReadStructure
========================

Subclass of the standard Geant4 G4GDMLReadStructure

   
Inheritance chain::

   X4GDMLReadStructure
   G4GDMLReadStructure
   G4GDMLReadParamvol
   G4GDMLReadSetup
   G4GDMLReadSolids
   G4GDMLReadMaterials
   G4GDMLReadDefine
   G4GDMLRead   

**/


#include "G4GDMLReadStructure.hh"

class X4_API X4GDMLReadStructure : public G4GDMLReadStructure 
{
    public:
        static const plog::Severity LEVEL ; 
    public:
        X4GDMLReadStructure() ; 
        const G4VSolid* read_solid(const char* path, int offset);  // offset:-1 last solid in store


};

