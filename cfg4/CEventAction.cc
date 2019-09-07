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

// g4-
#include "CFG4_PUSH.hh"
#include "G4Event.hh"
#include "CFG4_POP.hh"

// okc-
#include "Opticks.hh"

// cg4-
#include "CG4Ctx.hh"
#include "CG4.hh"
#include "CEventAction.hh"

#include "PLOG.hh"

/**
CEventAction
=================

Canonical instance (m_ea) is ctor resident of CG4 

**/

CEventAction::CEventAction(CG4* g4)
   : 
   G4UserEventAction(),
   m_g4(g4),
   m_ctx(g4->getCtx()),
   m_ok(g4->getOpticks())
{ 
}

CEventAction::~CEventAction()
{ 
}

void CEventAction::BeginOfEventAction(const G4Event* anEvent)
{
    OKI_PROFILE("CEventAction::BeginOfEventAction"); 
    setEvent(anEvent);
}

void CEventAction::EndOfEventAction(const G4Event* /*anEvent*/)
{
    OKI_PROFILE("CEventAction::EndOfEventAction"); 
}

void CEventAction::setEvent(const G4Event* event)
{
    m_ctx.setEvent(event);
}

void CEventAction::postinitialize()
{
    LOG(verbose) << "CEventAction::postinitialize" 
              << brief()
               ;
}

std::string CEventAction::brief()
{
    std::stringstream ss ; 
    ss  
       << " event_id " << m_ctx._event_id
       ;
    return ss.str();
}


