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

// CEventAction
// ================
//
//

#include <string>
#include "G4UserEventAction.hh"
#include "globals.hh"

class G4Event ; 

class Opticks ; 

struct CG4Ctx ; 
class CG4 ; 

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CEventAction : public G4UserEventAction
{
  public:
    virtual ~CEventAction();
  private:
    friend class CG4 ;    
    CEventAction(CG4* g4);
    void postinitialize();
  public:
    std::string brief(); 
  public:
    virtual void BeginOfEventAction(const G4Event* anEvent);
    virtual void EndOfEventAction(const G4Event* anEvent);
  private:
    void setEvent(const G4Event* event);
  private:
    CG4*               m_g4 ; 
    CG4Ctx&            m_ctx ; 
    Opticks*           m_ok ; 

};

#include "CFG4_TAIL.hh"

