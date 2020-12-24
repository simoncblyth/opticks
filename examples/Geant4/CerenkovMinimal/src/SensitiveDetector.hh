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

#include "G4VSensitiveDetector.hh"
#include "OpHit.hh"

class G4Step ; 
class G4TouchableHistory ;
class G4HCofThisEvent ; 


struct SensitiveDetector : public G4VSensitiveDetector
{
    static OpHitCollection* GetHitCollection(G4HCofThisEvent* HCE, const char* query );
    static void DumpHitCollections(G4HCofThisEvent* HCE);

    static const char* SDName ;
    static const char* collectionNameA ;
    static const char* collectionNameB ;

    SensitiveDetector(const char* name) ;  

    void Initialize(G4HCofThisEvent* HCE);
    void EndOfEvent(G4HCofThisEvent* HCE);
    G4bool ProcessHits(G4Step* aStep,G4TouchableHistory* ROhist);

    OpHitCollection* hitCollectionA ; 
    OpHitCollection* hitCollectionB ; 

    unsigned m_hit_count ;

};




