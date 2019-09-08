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


class G4ParticleDefinition ; 
class G4Track ; 
class G4Event ; 

#include "G4ThreeVector.hh"
#include "G4UserSteppingAction.hh"
#include "CBoundaryProcess.hh"
#include "globals.hh"

/**

CSteppingAction
================

Canonical instance (m_sa) is ctor resident of CG4.
CSteppingAction accepts steps from Geant4, routing them to either:

1. m_recorder(CRecorder) for optical photon steps
2. m_steprec(CStepRec) for non-optical photon steps

The setStep method returns a boolean "done" which
dictates whether to fStopAndKill the track.
The mechanism is used to stop tracking when reach truncation (bouncemax)
as well as absorption.

**/

class Opticks ; 

struct CG4Ctx ; 

class CG4 ; 
class CRandomEngine ; 
class CMaterialLib ; 
class CRecorder ; 
class CGeometry ; 
class CMaterialBridge ; 
class CStepRec ; 

class G4Navigator ; 
class G4TransportationManager ; 


#include "plog/Severity.h"

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CSteppingAction : public G4UserSteppingAction
{
   static const plog::Severity LEVEL ;  
   friend class CTrackingAction ; 

  public:
    CSteppingAction(CG4* g4, bool dynamic);
    void postinitialize();
    void report(const char* msg="CSteppingAction::report");
    virtual ~CSteppingAction();

  public:
    virtual void UserSteppingAction(const G4Step*);
  private:
    bool setStep( const G4Step* step);
    void prepareForNextStep( const G4Step* step);

  private:
    CG4*              m_g4 ; 
    CRandomEngine*    m_engine ; 
    CG4Ctx&           m_ctx ; 
    Opticks*          m_ok ; 
    bool              m_dbgflat ; 
    bool              m_dbgrec ; 
    bool              m_dynamic ; 
    CGeometry*        m_geometry ; 
    CMaterialBridge*  m_material_bridge ; 
    CMaterialLib*     m_mlib ; 
    CRecorder*        m_recorder   ; 
    CStepRec*         m_steprec   ; 

    G4TransportationManager* m_trman ; 
    G4Navigator*             m_nav ; 

    unsigned int m_steprec_store_count ;
    int          m_cursor_at_clear ;

};

#include "CFG4_TAIL.hh"

