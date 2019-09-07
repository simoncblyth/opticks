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

class Opticks ; 
class OpticksHub ; 
class CG4 ; 
class G4RunManager ; 

//#define OLDPHYS 1
#ifdef OLDPHYS
class PhysicsList ; 
#else
//class OpNovicePhysicsList ; 
class CPhysicsList ; 
#endif

#include "CFG4_API_EXPORT.hh"

/**
CPhysics
==========

HUH: why the runManager lives here ? , expected CG4

**/

class CFG4_API CPhysics 
{
    public:
        CPhysics(CG4* g4);
    public:
        G4RunManager* getRunManager() const ; 
        void setProcessVerbosity(int verbosity);
    private:
        int preinit();
        void init();
    private:
        CG4*           m_g4 ;     
        OpticksHub*    m_hub ;     
        Opticks*       m_ok ;     
        int            m_preinit ;   
        G4RunManager*  m_runManager ; 
#ifdef OLDPHYS
        PhysicsList*          m_physicslist ; 
#else
        //OpNovicePhysicsList*  m_physicslist ; 
        CPhysicsList*           m_physicslist ; 
#endif

};



