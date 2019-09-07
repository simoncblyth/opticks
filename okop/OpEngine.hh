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

class SLog ; 

class Opticks ;       // okc-
class OpticksEntry ; 
class OpticksHub ;    // okg-

class OScene ;   // optixrap-
class OPropagator ; 
class OEvent ; 
class OContext ; 

class OpSeeder ; 
class OpZeroer ; 
class OpIndexer ; 

#include "plog/Severity.h"
#include "OKOP_API_EXPORT.hh"

/**
OpEngine
=========

OpEngine takes a central role, it holds the OScene
which creates the OptiX context holding the GPU geometry
and all GPU buffers.

Instanciating an OpEngine 


Canonical OpEngine instance m_engine resides in ok-/OKPropagator 
which resides as m_propagator at top level in ok-/OKMgr

* BUT: ok- depends on OpenGL ... need compute only equivalents okop-/OpPropagator okop/OpMgr

NB OpEngine is ONLY AT COMPUTE LEVEL, FOR THE FULL PICTURE NEED TO SEE ONE LEVEL UP 
   IN ok-
   OKPropagator::uploadEvent 
   OKPropagator::downloadEvent

  
**/

class OKOP_API OpEngine {
       // friends can access the OPropagator
       friend class OpIndexer ; 
       friend class OpSeeder ; 
       friend class OpZeroer ; 
    public:
       static const plog::Severity LEVEL ;  
    public:
       OpEngine(OpticksHub* hub);
    public:
       OContext*    getOContext();         // needed by opticksgl-/OpViz

       void propagate();                // OPropagator prelaunch+launch : populates GPU photon, record and sequence buffers
       void indexEvent();
       unsigned downloadEvent();
       unsigned uploadEvent();
       unsigned getOptiXVersion();
       void cleanup();
       void Summary(const char* msg="OpEngine::Summary");

    private:
       OPropagator* getOPropagator();
    private:
       void downloadPhotonData();       // see App::dbgSeed
       int preinit() const ;
       void init();
       void initPropagation();
    private:
       // ctor instanciated always
       int                  m_preinit ; 
       SLog*                m_log ; 
       OpticksHub*          m_hub ; 
       Opticks*             m_ok ; 
       OScene*              m_scene ; 
       OContext*            m_ocontext ; 
    private:
       // conditionally instanciated in init, not for isLoad isTracer 
       OpticksEntry*        m_entry ; 
       OEvent*              m_oevt ; 
       OPropagator*         m_propagator ; 
       OpSeeder*            m_seeder ; 
       OpZeroer*            m_zeroer ; 
       OpIndexer*           m_indexer ; 
};


