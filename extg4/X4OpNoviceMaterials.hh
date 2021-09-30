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

/**

X4OpNoviceMaterials
====================

This struct creates materials for use only by tests, including:

1. extg4/tests/X4MaterialTest.cc 
2. extg4/tests/X4MaterialTableTest.cc

Due to a Geant4 API change the setting of spline interpolation to 
true has been removed from some the the properties.  
But no matter these materials are used just for machinery tests 
for which spline interpolation is irrelevant.


**/

#include "X4_API_EXPORT.hh"

class G4Material ;

struct X4_API X4OpNoviceMaterials
{
    G4Material* air ; 
    G4Material* water ;

    X4OpNoviceMaterials();
}; 
 



