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


#include <cstring>
#include "Opticks.hh"

#include "CG4.hh"
#include "CCerenkov.hh"
#include "G4Cerenkov.hh"
#include "G4Cerenkov1042.hh"
#include "C4Cerenkov1042.hh"

G4VProcess* CCerenkov::getProcess() const 
{ 
   return m_process ; 
}

CCerenkov::CCerenkov(const CG4* g4)
    :
    m_g4(g4),
    m_ok(g4->getOpticks()),
    m_class(m_ok->getCerenkovClass()),
    m_maxNumPhotonsPerStep(100),
    m_maxBetaChangePerStep(10),   // maximum allowed change in beta = v/c in % (perCent)
    m_trackSecondariesFirst(true), // photons first before resuming with primaries
    m_verboseLevel(1),
    m_process(initProcess(m_class))
{
} 

#define CONFIG_PROCESS(p)  \
   { \
         p->SetMaxNumPhotonsPerStep(m_maxNumPhotonsPerStep); \
         p->SetMaxBetaChangePerStep(m_maxBetaChangePerStep); \
         p->SetTrackSecondariesFirst(m_trackSecondariesFirst); \
         p->SetVerboseLevel(m_verboseLevel); \
   }

G4VProcess* CCerenkov::initProcess(const char* cls) const 
{
    G4VProcess* p = NULL ; 
    if( strcmp( cls, "G4Cerenkov" ) == 0 )
    {
         G4Cerenkov* s = new G4Cerenkov() ; 
         CONFIG_PROCESS(s);
         p = s ;    
    }
    else if( strcmp( cls, "G4Cerenkov1042" ) == 0 )
    {
         G4Cerenkov1042* s = new G4Cerenkov1042() ; 
         CONFIG_PROCESS(s);
         p = s ;   
    }
    else if( strcmp( cls, "C4Cerenkov1042" ) == 0 )
    {
         C4Cerenkov1042* s = new C4Cerenkov1042() ; 
         CONFIG_PROCESS(s);
         p = s ;   
    }
    return p ; 
}


