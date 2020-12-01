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
class SensorLib ; 

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

Notice that OpEngine is used by both *okop* and *ok* projects
leading to two canonical m_engine instanciations in the 
compute only *okop* and interop viz *ok* projects.

okop/OpPropagator.cc
   compute only branch using purely OptiX buffers, as used by 
   okop/OpMgr.cc(instanciated within G4Opticks::setGeometry) 

ok/OKPropagator.cc
   branch using underlying OpenGL buffers with OptiX referencing 
   them to enable interop visualization, as used by ok/OKMgr(OKTest) 
   and okg4/OKG4Mgr(OKG4Test)  

NB as OpEngine is ONLY AT COMPUTE LEVEL, FOR THE FULL PICTURE 
OF THE OK PROJECT OpenGL BUFFERS MUST SEE ONE LEVEL UP::

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
       OContext*    getOContext() const ;         // needed by opticksgl-/OpViz
       void uploadSensorLib(const SensorLib* sensorlib);

       void propagate();                // OPropagator prelaunch+launch : populates GPU photon, record and sequence buffers
       void indexEvent();
       unsigned downloadEvent();
       unsigned uploadEvent();
       unsigned getOptiXVersion() const ;
       void cleanup();
       void Summary(const char* msg="OpEngine::Summary");

    private:
       OPropagator* getOPropagator() const ;
    private:
       void downloadPhotonData();       // see App::dbgSeed
       int preinit() const ;
       void init();
       void initPropagation();
       void close(); 
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
       bool                 m_closed ;    
};

