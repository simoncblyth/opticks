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

struct STimes ; 
class SLog ; 
template <typename T> class NPY ;

class OpticksEvent ; 
class OpticksEntry ; 
class Opticks ; 
class OpticksHub ; 

#include "OXPPNS.hh"
#include "plog/Severity.h"

class OContext ; 
class ORng ; 
class OEvent ; 

/**
OPropagator : Launch control 
===============================

The name of this class is too similar to okop/OpPropagator.
TODO: rename to OLauncher 


Residents:

m_orng:ORng 
    hmm perhaps this should live in OScene, rather than here 


**/


#include "OXRAP_API_EXPORT.hh"
class OXRAP_API OPropagator {
    public:
        static const plog::Severity LEVEL ;   
    public:
        OPropagator( Opticks* ok, OEvent* oevt, OpticksEntry* entry); 
    public:
        void prelaunch();   
        void launch();
        std::string brief();
    public:
        void setOverride(unsigned int override);
        void setNoPropagate(bool nopropagate=true );
    private:
        void init();
        void setEntry(unsigned int entry);
        void initParameters();
        void resize(); 
        void setSize(unsigned width, unsigned height);
    private:
        SLog*                m_log ; 
        Opticks*             m_ok ; 
        OEvent*              m_oevt ; 
        OContext*            m_ocontext ; 
        optix::Context       m_context ;
        ORng*                m_orng ; 

        int                  m_propagateoverride ; 
        bool                 m_nopropagate ; 

        OpticksEntry*        m_entry ; 
        int                  m_entry_index ; 

        bool                 m_prelaunch ;

    private:
        unsigned             m_prelaunch_count ; 
        unsigned             m_launch_count ; 
        unsigned             m_width ; 
        unsigned             m_height ; 
        unsigned             m_launch_acc ; 
        unsigned             m_launch_lis ; 

 
};


