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

#include "CRunAction.hh"
#include "CManager.hh"
#include "Opticks.hh"
#include "PLOG.hh"

const plog::Severity CRunAction::LEVEL = PLOG::EnvLevel("CRunAction", "DEBUG"); 

CRunAction::CRunAction(CManager* manager) 
   :
     G4UserRunAction(),
     m_manager(manager),
     m_count(0)
{
    LOG(LEVEL) << "count " << m_count   ;
}
CRunAction::~CRunAction()
{
    LOG(LEVEL) << "count " << m_count  ;
}
void CRunAction::BeginOfRunAction(const G4Run* run)
{
    OKI_PROFILE("CRunAction::BeginOfRunAction");
    m_count += 1 ; 
    m_manager->BeginOfRunAction(run); 
    LOG(LEVEL) << "count " << m_count  ;
}
void CRunAction::EndOfRunAction(const G4Run* run)
{
    OKI_PROFILE("CRunAction::EndOfRunAction"); 
    m_manager->EndOfRunAction(run); 
    LOG(LEVEL) << "count " << m_count  ;
}



