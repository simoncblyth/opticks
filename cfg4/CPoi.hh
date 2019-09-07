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
class G4StepPoint ; 

#include "CStage.hh"
#include "CBoundaryProcess.hh"

#include "CFG4_API_EXPORT.hh"

class CFG4_API CPoi 
{
    public:
#ifdef USE_CUSTOM_BOUNDARY
        Ds::DsG4OpBoundaryProcessStatus getBoundaryStatus() const ;
        CPoi(const G4StepPoint* point, unsigned flag, unsigned material, Ds::DsG4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage, const G4ThreeVector& origin);
#else
        G4OpBoundaryProcessStatus getBoundaryStatus() const  ;
        CPoi(const G4StepPoint* point, unsigned flag,  unsigned material, G4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage, const G4ThreeVector& origin);
#endif
        ~CPoi(); 

        const G4StepPoint* getPoint() const ; 
        unsigned           getFlag() const ; 
        unsigned           getMaterial() const ; 
        CStage::CStage_t   getStage() const ;
        std::string        description() const ; 

    private: 
        const G4StepPoint*  m_point ; 
        unsigned            m_flag ; 
        unsigned            m_material ; 


#ifdef USE_CUSTOM_BOUNDARY
        Ds::DsG4OpBoundaryProcessStatus m_boundary_status ;
#else
        G4OpBoundaryProcessStatus   m_boundary_status ;
#endif
        CStage::CStage_t  m_stage ;
        int               m_action ; 
        G4ThreeVector     m_origin ;

};
