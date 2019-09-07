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


#include "CFG4_PUSH.hh"
// huh: G4OpBoundaryProcess pulls in CLHEP/Random headers that raise warnings
#include "CBoundaryProcess.hh"
#include "CStage.hh"
#include "CFG4_POP.hh"

class G4Step ; 
class G4StepPoint ; 


#include "CFG4_API_EXPORT.hh"
class CFG4_API State 
{
   public:
#ifdef USE_CUSTOM_BOUNDARY
       State(const G4Step* step, Ds::DsG4OpBoundaryProcessStatus boundary_status, unsigned int premat, unsigned int postmat, CStage::CStage_t stage, unsigned action);
       Ds::DsG4OpBoundaryProcessStatus getBoundaryStatus() const ;
#else
       State(const G4Step* step, G4OpBoundaryProcessStatus boundary_status, unsigned int premat, unsigned int postmat, CStage::CStage_t stage, unsigned action);
       G4OpBoundaryProcessStatus getBoundaryStatus() const ;
#endif
       virtual ~State();
   public:
       const G4Step*      getStep() const ;  
       const G4StepPoint* getPreStepPoint() const ;
       const G4StepPoint* getPostStepPoint() const ;
       unsigned int       getPreMaterial() const ;
       unsigned int       getPostMaterial() const ;
       CStage::CStage_t   getStage() const ; 
       unsigned           getAction() const ; 
   private:
       const G4Step*             m_step ;
#ifdef USE_CUSTOM_BOUNDARY
       Ds::DsG4OpBoundaryProcessStatus m_boundary_status ;
#else
       G4OpBoundaryProcessStatus m_boundary_status ;
#endif
       unsigned                  m_premat ;
       unsigned                  m_postmat ;
       CStage::CStage_t          m_stage ; 
       unsigned                  m_action ; 
};



