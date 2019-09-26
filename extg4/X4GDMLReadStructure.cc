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

#include "X4GDMLReadStructure.hh"
#include "X4SolidStore.hh"

#include <cstring>
#include "PLOG.hh"

const plog::Severity X4GDMLReadStructure::LEVEL = PLOG::EnvLevel("X4GDMLReadStructure", "DEBUG" ); 

X4GDMLReadStructure::X4GDMLReadStructure()
{
}

const G4VSolid* X4GDMLReadStructure::read_solid(const char* path )
{
     G4String fileName = path ; 
     G4bool validation = false ; 
     G4bool isModule = false ; 
     G4bool strip = false ;  

     Read( fileName, validation, isModule, strip ); 

     return X4SolidStore::Get(-1) ;    // last solid in store
}



