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

struct NSnapConfig ; 

class SLog ; 

// okc-
class Composition ; 

// okg-
class OpticksHub ;

// optixrap-
class OContext ;
class OTracer ;

//opop-
class OpEngine ; 

template <typename T> class NPY ;


#include "plog/Severity.h"
#include "OKOP_API_EXPORT.hh"
#include "SRenderer.hh"

/**
OpTracer
=========

Canonical m_tracer instance is resident of OpPropagator
and is instanciated with it.

snap() takes a sequence of ppm geomtry snapshots 
configured via --snapconfig and switched on via --snap
see okop-


**/




class OKOP_API OpTracer : public SRenderer {
    public:
       static const plog::Severity LEVEL ;  
    public:
       OpTracer(OpEngine* ope, OpticksHub* hub, bool immediate);
    public:
       void snap(const char* dir, const char* reldir=NULL);
    private:
       static int Preinit();
       void init();
       void initTracer();
       void multi_snap(const char* dir, const char* reldir=NULL);
       void multi_snap(const char* path_fmt, const std::vector<glm::vec3>& eyes ) ;
       void single_snap(const char* path);

       void render();     // fulfils SRenderer protocol
       void setup_render_target() ;
    private:
       int              m_preinit ; 
       OpEngine*        m_ope ; 
       OpticksHub*      m_hub ; 
       Opticks*         m_ok ; 
       NSnapConfig*     m_snap_config ; 
       bool             m_immediate ; 

       OContext*        m_ocontext ; 
       Composition*     m_composition ; 
       OTracer*         m_otracer ;
       unsigned         m_count ; 

};


