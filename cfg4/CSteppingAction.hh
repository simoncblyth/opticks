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


**/

class Opticks ; 

struct CManager ; 

#include "plog/Severity.h"

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CSteppingAction : public G4UserSteppingAction
{
   static const plog::Severity LEVEL ;  
   friend class CTrackingAction ; 

  public:
    CSteppingAction(CManager* manager);
    virtual ~CSteppingAction();
  public:
    virtual void UserSteppingAction(const G4Step*);
  private:
    CManager* m_manager ;  


};

#include "CFG4_TAIL.hh"

