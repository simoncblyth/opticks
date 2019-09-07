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

#include <string>

#include "G4ThreeVector.hh" 
class G4Step ; 

#include "CStage.hh"
#include "CBoundaryProcess.hh"

#include "CFG4_API_EXPORT.hh"

class CFG4_API CStp 
{
   public:
#ifdef USE_CUSTOM_BOUNDARY
       Ds::DsG4OpBoundaryProcessStatus getBoundaryStatus() ;
       CStp(const G4Step* step, int step_id, Ds::DsG4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage, const G4ThreeVector& origin);
       CStp(const G4Step* step, int step_id, Ds::DsG4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, CStage::CStage_t stage, int action, const G4ThreeVector& origin);
#else
       G4OpBoundaryProcessStatus   getBoundaryStatus() ;
       CStp(const G4Step* step, int step_id,   G4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage, const G4ThreeVector& origin);
       CStp(const G4Step* step, int step_id,   G4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, CStage::CStage_t stage, int action, const G4ThreeVector& origin); 
#endif
       ~CStp();  

        std::string description();
        std::string origin();

    public:
         const G4Step*    getStep();
         int              getStepId(); 
         CStage::CStage_t getStage();

         void   setMat(  unsigned premat, unsigned postmat);
         void   setFlag( unsigned preflag, unsigned postflag );
         void   setAction(int action);

   private:
         const G4Step*               m_step ; 
         int                         m_step_id ; 
#ifdef USE_CUSTOM_BOUNDARY
         Ds::DsG4OpBoundaryProcessStatus m_boundary_status ;
#else
         G4OpBoundaryProcessStatus   m_boundary_status ;
#endif
         unsigned          m_premat ; 
         unsigned          m_postmat ; 
         unsigned          m_preflag ; 
         unsigned          m_postflag ; 

         CStage::CStage_t  m_stage ;
         int               m_action ; 
         G4ThreeVector     m_origin ;

};

