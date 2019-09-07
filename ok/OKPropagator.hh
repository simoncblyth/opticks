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
template <typename T> class NPY ; 

class OpticksHub ; 
class Opticks ; 
class OpticksIdx ; 
class OpticksViz ; 
class Opticks ; 


#ifdef OPTICKS_OPTIX
class OpEngine ; 
class OKGLTracer ; 
#endif

#include "OK_API_EXPORT.hh"
#include "OK_HEAD.hh"
#include "plog/Severity.h"

/**
OKPropagator
===============

Perform GPU propagation of event 
currently lodged in hub. This incorporates
uploading the event gensteps to GPU, 
doing the OptiX launch to populate 
the buffers and downloading back into the 
event.

Methods intended to operate above the 
level of the compute/interop split.

* core functionality of this could be down in okop ?

**/

class OK_API OKPropagator {
   public:
       static const plog::Severity LEVEL ; 
       static OKPropagator* GetInstance();
   public:
       OKPropagator(OpticksHub* hub, OpticksIdx* idx, OpticksViz* viz);
   private:
       int preinit() const ; 
       void init();  
   public:
       void propagate();
       void cleanup();
   public:
       int uploadEvent();
       int downloadEvent();
       void indexEvent();
   private:
       static OKPropagator* fInstance ;  
       int            m_preinit ; 
       SLog*          m_log ; 
       OpticksHub*    m_hub ; 
       OpticksIdx*    m_idx ; 
       OpticksViz*    m_viz ; 
       Opticks*       m_ok ; 
#ifdef OPTICKS_OPTIX
       OpEngine*      m_engine ; 
       OKGLTracer*    m_tracer ; 
#endif
       int            m_placeholder ; 
       
};

#include "OK_TAIL.hh"


