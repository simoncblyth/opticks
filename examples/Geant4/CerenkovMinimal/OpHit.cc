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

#include "OpHit.hh"

OpHit* OpHit::MakeDummyHit()
{  
    OpHit* hit = new OpHit ; 
    hit->ene = 1. ; 
    hit->tim = 1. ; 
    hit->pos = G4ThreeVector(1,1,1); 
    hit->dir = G4ThreeVector(0,0,1); 
    hit->pol = G4ThreeVector(1,0,0); 

    return hit ; 
}


