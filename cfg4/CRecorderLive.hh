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

// **CURRENTLY DEAD CODE**

// fork(or have both for crosscheck) at higher CG4 level 
//   m_live(m_ok->hasOpt("liverecorder")),


/**

CRecorderLive 
=================

live recording mode : **currently not used**
---------------------------------------------

* much of the code for this currently parked in CRecorderDead.cc

OpticksEvent records written during stepping.
*LiveRecordStep* is called for all G4Step
each of which is comprised of *pre* and *post* G4StepPoint, 
as a result the same G4StepPoint are "seen" twice, 
thus *RecordStep* only records the 1st of the pair 
(the 2nd will come around as the first at the next call)
except for the last G4Step pair where both points are recorded

*photons_per_g4event* is used by defineRecordId so the different
technical g4 events all get slotted into the same OpticksEvent record 
buffers

**/


#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CRecorderLive {
        friend class CSteppingAction ;
   public:
        static const char* PRE ; 
        static const char* POST ; 
   public:
        CRecorderLive(CG4* g4, CGeometry* geometry, bool dynamic);
        void postinitialize();               // called after G4 geometry constructed in CG4::postinitialize
        void initEvent(OpticksEvent* evt);   // called prior to recording, sets up writer (output buffers)
        CRec* getCRec() const ; 
   private:
        void setEvent(OpticksEvent* evt);
   public:
        void posttrack();                    // invoked from CTrackingAction::PostUserTrackingAction for optical photons
   public:
        void zeroPhoton();
   public:

#ifdef USE_CUSTOM_BOUNDARY
    public:
        bool Record(DsG4OpBoundaryProcessStatus boundary_status);
    private:
        bool RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, DsG4OpBoundaryProcessStatus boundary_status, const char* label);
        void setBoundaryStatus(DsG4OpBoundaryProcessStatus boundary_status, unsigned int preMat, unsigned int postMat);
#else
    public:
        bool Record(G4OpBoundaryProcessStatus boundary_status);
    private:
        bool RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, const char* label);
        void setBoundaryStatus(G4OpBoundaryProcessStatus boundary_status, unsigned int preMat, unsigned int postMat);
#endif
    private:
        void setStep(const G4Step* step, int step_id);
        //bool LiveRecordStep();
        void RecordStepPoint(unsigned int slot, const G4StepPoint* point, unsigned int flag, unsigned int material, const char* label);
        //void Clear();
   public:
        std::string getStepActionString();
   public:
        // for reemission continuation
        void decrementSlot();
   public:
        void posttrackWriteSteps();
   public:
        void Summary(const char* msg);
        std::string desc() const ; 
        void dump(const char* msg="CRecorder::dump");
   private:
        CG4*               m_g4; 
        CG4Ctx&            m_ctx; 
        Opticks*           m_ok; 
        CPhoton            m_photon ;  
        CRecState          m_state ;  
        CRec*              m_crec ; 
        CDebug*            m_dbg ; 

        OpticksEvent*      m_evt ; 
        CGeometry*         m_geometry ; 
        CMaterialBridge*   m_material_bridge ; 
        bool               m_dynamic ;
        bool               m_live ;   
        CWriter*           m_writer ; 


    private:


        // below are zeroed in zeroPhoton
#ifdef USE_CUSTOM_BOUNDARY
        DsG4OpBoundaryProcessStatus m_boundary_status ; 
        DsG4OpBoundaryProcessStatus m_prior_boundary_status ; 
#else
        G4OpBoundaryProcessStatus m_boundary_status ; 
        G4OpBoundaryProcessStatus m_prior_boundary_status ; 
#endif
        unsigned m_premat ; 
        unsigned m_prior_premat ; 

        unsigned m_postmat ; 
        unsigned m_prior_postmat ; 





};
#include "CFG4_TAIL.hh"

