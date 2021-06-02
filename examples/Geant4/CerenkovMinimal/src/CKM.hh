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

class G4RunManager ;
class G4VPhysicalVolume ; 

class  G4Opticks ; 
struct G4OpticksRecorder ; 

struct RAction ; 
struct EAction ; 
struct TAction ; 
struct SAction ; 

struct SensitiveDetector ;
struct DetectorConstruction ;
class L4Cerenkov ; 
class CKMScintillation ; 

template <typename C, typename S> struct PhysicsList ; 
struct PrimaryGeneratorAction ;


struct CKM
{
    CKM(); 
    ~CKM();

    void init(); 
    void beamOn(int nev);
    void setup_G4Opticks(const G4VPhysicalVolume* world );

    G4Opticks*               g4ok ; 
    G4OpticksRecorder*       okr ; 
    G4RunManager*            rm ; 
    const char*             sdn ; 
    SensitiveDetector*       sd ; 
    DetectorConstruction*    dc ; 
    PhysicsList<L4Cerenkov,CKMScintillation>* pl ;
    PrimaryGeneratorAction*  ga ; 

    RAction* ra ; 
    EAction* ea ; 
    TAction* ta ; 
    SAction* sa ; 

}; 


