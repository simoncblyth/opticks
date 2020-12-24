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

struct Ctx ; 
struct SensitiveDetector ;
struct DetectorConstruction ;
class L4Cerenkov ; 
template <typename T> struct PhysicsList ; 
struct PrimaryGeneratorAction ;

struct RunAction ; 
struct EventAction ; 
struct TrackingAction ; 
struct SteppingAction ; 

struct G4
{
    G4(int nev); 
    ~G4();
    void beamOn(int nev);

    Ctx*                    ctx ; 
    G4RunManager*            rm ; 
    const char*             sdn ; 
    SensitiveDetector*       sd ; 
    DetectorConstruction*    dc ; 
    PhysicsList<L4Cerenkov>* pl ;
    PrimaryGeneratorAction*  ga ; 
    RunAction*               ra ; 
    EventAction*             ea ; 
    TrackingAction*          ta ; 
    SteppingAction*          sa ; 
}; 


