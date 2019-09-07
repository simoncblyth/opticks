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

#include <map>
#include <string>
class CG4 ; 
struct CG4Ctx ; 
class CMaterialLib ; 

//#include "G4ThreeVector.hh"
#include "G4Transportation.hh"

class DebugG4Transportation : public G4Transportation 
{
   public:
       DebugG4Transportation( CG4* g4, G4int verbosityLevel= 1); 
   private:
       void init();
   public:
       std::string firstMaterialWithGroupvelAt430nm(float groupvel, float delta=0.001f);
       G4VParticleChange* AlongStepDoIt( const G4Track& track, const G4Step&  stepData );

   private:
       CG4*          m_g4 ; 
       CG4Ctx&       m_ctx ; 
       CMaterialLib* m_mlib ; 
       G4ThreeVector m_origin ; 


};




