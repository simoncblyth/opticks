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

// g4-
class G4Step ; 

#include "CFG4_PUSH.hh"
#include "CBoundaryProcess.hh"
#include "CFG4_POP.hh"

class Opticks ; // okc-
class OpticksEvent ; 

// cfg4-
struct CCtx ; 
#include "CPhoton.hh"
#include "CRecState.hh"
#include "plog/Severity.h"

class CDebug ; 
class CRec ; 
class CMaterialBridge ; 
class CWriter ; 
class CStp ; 
class CGeometry ; 

/**
CRecorder : records Geant4 optical photons into OpticksEvent arrays
=====================================================================

Canonical m_recorder instance is ctor resident of CManager.
Steps are supplied to CRecorder via two different routes.

With OKG4 (non-embedded Opticks+Geant4 running) from::

    CSteppingAction::UserSteppingAction
    CManager::UserSteppingAction

With embedded running::

    G4OpticksRecorder::UserSteppingAction
    CManager::UserSteppingAction
    

The objective of *CRecorder* is to collect Geant4 photon 
steps in a format that precisely matches the Opticks GPU 
photon records allowing use of the Opticks analysis 
and visualization tools with G4 simulations.

To this end *CRecorder* saves non-dynamically into buffer of
fixed number of photons and max steps per photon 
in order to match on-GPU restrictions.  setQuad with
a computed record_id and slot_id is used to mimick
separate CUDA thread writes into tranches of record buffer. 

CRecorder uses canned recording style, where steps are 
collected with *Record* and then only written out to 
OpticksEvent at *posttrackWriteSteps*.
Note that CRecorderLive (currently dead) uses step-by-step writing.


Principal CRecorder call routes
-----------------------------------

CManager::BeginOfEventAction -> CManager::presave -> CManager::initEvent -> CRecorder::initEvent
    prepares m_writer CWriter::initEvent m_crec CRec::initEvent

CManager::setStep   -> CRecorder::Record
    prime driver 

CManager::PostUserTrackingAction -> CManager::postTrack -> CRecorder::postTrack
    In default --recstp mode CRecorder::postTrackWriteSteps
    passes the m_crec stp collected by CRec to CWriter with CRecorder::WriteStepPoint  


Example
---------

Consider 
   TO RE BT BT BT BT SA

Live mode:
   write pre until last when write pre,post 

Canned mode:
    For first write pre,post then write post

Rejoins are not known until another track comes along 
that lines up with former ending in AB. 


CRecorder::posttrackWriteSteps
--------------------------------

Although much better now, tis still complicated with loadsa special cases.

Possibly can drastically simplify (and make much closer to generate.cu) 
by step-by-step collecting G4StepPoint (with skips done) rather than 
the current collecting of G4Step.

Fixing notes/issues/cfg4-bouncemax-not-working.rst required ~doubling 
the step limit as G4 "BR" "StepTooSmall" turnarounds really burn 
thru steps.


Debugging
-------------

--dbgseqhis 0x3ccc1
    switch on debug output for photons with a particular history, 
    showing steps, points, flags, volume names  
    get the seqhis hexstring from eg ana/evt.py:evt.seqhis_ana.table 

--dbgseqmat 0x11232
    switch on debug output for photons with a particular material history

--dbgrec
    machinery debugging, only useful for dumping machinery actions with 
    a small number of photons

**/

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CRecorder {
    public:
        static const plog::Severity LEVEL ; 
        static const char* PRE ; 
        static const char* POST ; 
    public:
        std::string        getStepActionString();
        CRec*              getCRec() const ; 
        unsigned long long getSeqHis() const ;
        unsigned long long getSeqMat() const ;

    public:
        CRecorder(CCtx& ctx); // CG4::CG4

        void setMaterialBridge(const CMaterialBridge* material_bridge);

        void initEvent(OpticksEvent* evt);   // called prior to recording, sets up writer (output buffers)
        void BeginOfGenstep();

        void postTrack();                    // invoked from CTrackingAction::PostUserTrackingAction for optical photons
#ifdef USE_CUSTOM_BOUNDARY
        bool Record(Ds::DsG4OpBoundaryProcessStatus boundary_status);
#else
        bool Record(G4OpBoundaryProcessStatus boundary_status);
#endif
    private:
        void compareModes(); 
        void zeroPhoton();
        void decrementSlot(); // for reemission continuation, which changes terminating AB into RE 

        void postTrackWriteSteps();  // using CStp 
        void postTrackWritePoints();  // experimental alternative using CPoi
 
#ifdef USE_CUSTOM_BOUNDARY
        bool WriteStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, Ds::DsG4OpBoundaryProcessStatus boundary_status, const char* label, bool last);
#else
        bool WriteStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, const char* label, bool last );
#endif
        void checkTopSlotRewrite(unsigned flag);

    private:
         // debugging 
        void pointDump( const char* msg, const G4StepPoint* point ) const ; 
    public:
        void Summary(const char* msg);
        void dump(const char* msg="CRecorder::dump");
        std::string desc() const ; 
        std::string brief() const ; 
    private:
        CCtx&            m_ctx; 
        Opticks*           m_ok; 
        double             m_microStep_mm ; 
        bool               m_suppress_same_material_microStep ; 
        unsigned           m_mode ; 
        bool               m_recpoi ; 
        bool               m_reccf ;
        CRecState          m_state ;  
        CPhoton            m_photon ;  
        CRec*              m_crec ; 
        CDebug*            m_dbg ; 

        OpticksEvent*      m_evt ; 
        CGeometry*         m_geometry ; 
        const CMaterialBridge*   m_material_bridge ; 
        bool               m_live ;   
        CWriter*           m_writer ; 
        unsigned           m_not_done_count ; 

        //unsigned           m_postTrack_acc ; 
  

};
#include "CFG4_TAIL.hh"
