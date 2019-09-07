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

#include "G4RunManager.hh"


#ifdef OLDPHYS
#include "PhysicsList.hh"
#else
#include "OpNovicePhysicsList.hh"
#endif

#include "Opticks.hh"
#include "OpticksHub.hh"
#include "CG4.hh"
#include "CPhysics.hh"
#include "CPhysicsList.hh"

int CPhysics::preinit()
{
    OK_PROFILE("_CPhysics::CPhysics"); 
    return 0 ; 
}

CPhysics::CPhysics(CG4* g4) 
    :
    m_g4(g4),
    m_hub(g4->getHub()),
    m_ok(g4->getOpticks()),
    m_preinit(preinit()),
    m_runManager(new G4RunManager),
#ifdef OLDPHYS
    m_physicslist(new PhysicsList())
#else
    m_physicslist(new CPhysicsList(m_g4))
    //m_physicslist(new OpNovicePhysicsList(m_g4))
#endif
{
    init();
}

void CPhysics::init()
{
    OK_PROFILE("CPhysics::CPhysics"); 
    m_runManager->SetNumberOfEventsToBeStored(0); 
    m_runManager->SetUserInitialization(m_physicslist);
}

G4RunManager* CPhysics::getRunManager() const 
{
   return m_runManager ; 
}

void CPhysics::setProcessVerbosity(int verbosity)
{
   // NB processes are instanciated only after PhysicsList Construct that happens at runInitialization 
   // so this needs to be called after then
#ifdef OLDPHYS
#else
    m_physicslist->setProcessVerbosity(verbosity); 
#endif
}



