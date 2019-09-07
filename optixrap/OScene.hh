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
class BTimeKeeper ; 
class Opticks ;
class OpticksHub ;
template <typename> class OpticksCfg ;

class OContext ; 
class OFunc ; 
class OColors ; 
class OGeo ; 
class OBndLib ; 
class OScintillatorLib ; 
class OSourceLib ; 

/**
OScene
========

Canonical m_scene instance resides in okop-/OpEngine 

Instanciating an OScene creates the OptiX GPU context 
and populates it with geometry, boundary info etc.. 
effectively uploading the geometry obtained from
the OpticksHub to the GPU.  This geometry info is 
held in the O* libs: OGeo, OBndLib, OScintillatorLib, 
OSourceLib.

NB there is no use of OptiX types in this interface header
although these are used internally. This is as are aiming 
to remove OptiX dependency in higher level interfaces 
for easier OptiX version hopping.

**/

#include "plog/Severity.h"
#include "OXRAP_API_EXPORT.hh"
class OXRAP_API OScene {
    public:
       static const plog::Severity LEVEL ; 
    public:
       OScene(OpticksHub* hub, const char* cmake_target="OptiXRap", const char* ptxrel=nullptr); 
    public:
       OContext*    getOContext();
       OBndLib*     getOBndLib();
    public:
       void cleanup();
    private:
       int preinit() const ; 
       void init();   // creates OptiX context and populates with geometry info
    private:
       int                  m_preinit ; 
       SLog*                m_log ; 
       BTimeKeeper*         m_timer ;
       OpticksHub*          m_hub ; 
       Opticks*             m_ok ; 

       OContext*         m_ocontext ; 
       OFunc*            m_osolve ; 
       OColors*          m_ocolors ; 
       OGeo*             m_ogeo ; 
       OBndLib*          m_olib ; 
       OScintillatorLib* m_oscin ; 
       OSourceLib*       m_osrc ; 
       unsigned          m_verbosity ; 
       bool              m_use_osolve ; 

};

