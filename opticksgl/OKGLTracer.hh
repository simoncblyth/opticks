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

// okc-
class Composition ; 

// okg-
class OpticksHub ;

// optixrap-
class OContext ;
class OTracer ;

//opop-
class OpEngine ; 

// optixgl-
class OFrame ;
class ORenderer ;

// oglrap-
class Scene ; 
class Interactor ; 
class OpticksViz ; 


#include "OKGL_API_EXPORT.hh"

#include "plog/Severity.h"
#include "SRenderer.hh"

/**
OKGLTracer
============

Establishes OpenGL interop between oxrap.OTracer and oglrap.Scene/Renderer

Canonical m_tracer instance is a resident of ok.OKPropagator 
when visualization is enabled (m_viz).

SRenderer protocol base, just: "void render()"
**/


class OKGL_API OKGLTracer : public SRenderer {
       static const plog::Severity LEVEL ; 
    public:
       static OKGLTracer* GetInstance();
    public:
       OKGLTracer(OpEngine* ope, OpticksViz* viz, bool immediate);
    public:
       void prepareTracer();
       void render();     // fulfils SRenderer protocol
    private:
       void init();
    private:
       static OKGLTracer* fInstance ; 
       SLog*            m_log ; 
       OpEngine*        m_ope ; 
       OpticksViz*      m_viz ; 
       OpticksHub*      m_hub ; 
       bool             m_immediate ; 
       Scene*           m_scene ;

       OContext*        m_ocontext ; 
       Composition*     m_composition ; 
       Interactor*      m_interactor ;
       OFrame*          m_oframe ;
       ORenderer*       m_orenderer ;
       OTracer*         m_otracer ;
 
       unsigned         m_trace_count ; 

};


