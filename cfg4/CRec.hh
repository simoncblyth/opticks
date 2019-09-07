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

#include <vector>
#include "plog/Severity.h"
#include "G4ThreeVector.hh"

#include "CBoundaryProcess.hh" 
#include "CStage.hh"

class Opticks ; 
class OpticksEvent ; 
class CStp ; 
class CPoi ; 
class CG4 ; 
class CMaterialBridge ; 

struct CRecState ; 
struct CG4Ctx ; 

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/**
CRec
=====

Canonical m_crec instance is resident of CRecorder and is instanciated with it.

**/

class CFG4_API CRec 
{
        static const plog::Severity LEVEL ;  
    public:
        CRec(CG4* g4, CRecState& state);
        void initEvent(OpticksEvent* evt);

        bool is_step_limited() const ; 

/*
        bool is_limited() const ; 
        bool is_point_limited() const ;    // relevant to recpoi mode only
*/
        std::string desc() const ;

        void setOrigin(const G4ThreeVector& origin);
        void clear();
        void setMaterialBridge(CMaterialBridge* material_bridge) ;

        void dump(const char* msg="CRec::dump");

        unsigned getNumStp() const ;
        CStp* getStp(unsigned index) const ;

        unsigned getNumPoi() const ;
        CPoi* getPoi(unsigned index) const ;
        CPoi* getPoiLast() const ;

   public:

#ifdef USE_CUSTOM_BOUNDARY
        bool add(Ds::DsG4OpBoundaryProcessStatus boundary_status);
#else
        bool add(G4OpBoundaryProcessStatus boundary_status);
#endif

   private:
        bool addPoi(CStp* stp);
        bool addPoi_(CPoi* poi);

#ifdef USE_CUSTOM_BOUNDARY
        void setBoundaryStatus(Ds::DsG4OpBoundaryProcessStatus boundary_status);
#else
        void setBoundaryStatus(G4OpBoundaryProcessStatus boundary_status);
#endif


#ifdef USE_CUSTOM_BOUNDARY
        void add(Ds::DsG4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, int action);
#else
        void add(G4OpBoundaryProcessStatus boundary_status, unsigned premat, unsigned postmat, unsigned preflag, unsigned postflag, int action);
#endif
    private:
        CG4*                        m_g4 ; 
        CRecState&                  m_state ; 
        CG4Ctx&                     m_ctx ; 

        Opticks*                    m_ok ; 
        bool                        m_recpoi ; 
        bool                        m_recpoialign ; 

        bool                        m_step_limited ; 
        bool                        m_point_done ; 

        CMaterialBridge*            m_material_bridge ; 
    private:
        G4ThreeVector               m_origin ; 
        std::vector<CStp*>          m_stp ; 
        std::vector<CPoi*>          m_poi ; 

   private:
        //unsigned                   m_add_acc ;   
#ifdef USE_CUSTOM_BOUNDARY
        Ds::DsG4OpBoundaryProcessStatus m_prior_boundary_status ; 
        Ds::DsG4OpBoundaryProcessStatus m_boundary_status ; 
#else
        G4OpBoundaryProcessStatus m_prior_boundary_status ; 
        G4OpBoundaryProcessStatus m_boundary_status ; 
#endif


};

#include "CFG4_TAIL.hh"

