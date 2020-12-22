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
        static const G4VSolid* ReadSolid(const char* path); 
        static const G4VSolid* ReadSolidFromString(const char* gdmlstring); 
        static void WriteGDMLString(const char* gdmlstring, const char* path) ;
        static const char* WriteGDMLStringToTmpPath(const char* gdmlstring);
    public:
        X4GDMLReadStructure() ; 
        const G4VSolid* getSolid(int offset=-1);
        void readString(const char* gdmlstring); 
        void readFile(const char* path); 
        void dumpMatrixMap(const char* msg="X4GDMLReadStructure::dumpMatrixMap") const ;


};

