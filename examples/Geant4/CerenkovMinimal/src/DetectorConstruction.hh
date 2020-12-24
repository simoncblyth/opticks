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

#include "G4VUserDetectorConstruction.hh"
#include "G4MaterialPropertyVector.hh"

class G4Material ; 
class G4VPhysicalVolume ;

struct DetectorConstruction : public G4VUserDetectorConstruction
{
    static G4Material* MakeAir(); 
    static G4Material* MakeWater(); 
    static G4Material* MakeGlass(); 
    static G4MaterialPropertyVector* MakeAirRI() ;
    static G4MaterialPropertyVector* MakeWaterRI() ;
    static G4MaterialPropertyVector* MakeGlassRI() ;
    static G4MaterialPropertyVector* MakeConstantProperty(float value);

    static void AddProperty( G4Material* mat , const char* name, G4MaterialPropertyVector* mpv );

    DetectorConstruction( const char* sdname_ ); 
    const char* sdname ;     

    virtual G4VPhysicalVolume* Construct();
};

