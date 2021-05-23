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


// cg4-
#include "CManager.hh"
#include "CEventAction.hh"
#include "PLOG.hh"

/**
CEventAction
=================

Canonical instance (m_ea) is ctor resident of CG4 

**/

CEventAction::CEventAction(CManager* manager)
   : 
   G4UserEventAction(),
   m_manager(manager)
{ 
}

CEventAction::~CEventAction()
{ 
}

void CEventAction::BeginOfEventAction(const G4Event* event)
{
    m_manager->BeginOfEventAction(event); 
}

void CEventAction::EndOfEventAction(const G4Event* event)
{
    m_manager->EndOfEventAction(event); 
}

