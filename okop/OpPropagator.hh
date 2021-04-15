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

template <typename T> class NPY ; 

class SensorLib ; 
class OpticksHub ; 
class Opticks ; 
class OpticksIdx ; 
class Opticks ; 

class OpEngine ; 
class OpTracer ; 

#include "OKOP_API_EXPORT.hh"
#include "OKOP_HEAD.hh"
#include "plog/Severity.h"

/**
OpPropagator : compute only propagator, no viz
==================================================

OpPropagator only used from OpMgr as m_propagator, which is used in
the G4Opticks approach, ie Opticks embedded inside an unsuspecting 
G4 example.   

Residents which are instanciated in ctor:

m_engine:OpEngine :
   control of GPU optical photon propagation

m_tracer:OpTracer 
   can make sequences of raytrace snapshots of geometry
   which can be saved to PPM files for subsequent conversion
   into PNG images or MP4 movies 


DevNotes
----------

Contrast with the viz enabled ok/OKPropagator

**/


class OKOP_API OpPropagator {
   public:
       static const plog::Severity LEVEL ; 
   public:
       OpPropagator(OpticksHub* hub, OpticksIdx* idx );
   public:
       //void uploadSensorLib(const SensorLib* sensorlib); 
   public:
       void propagate();
       void cleanup();
       void snap(const char* dir, const char* reldir=NULL);
       void flightpath(const char* dir, const char* reldir=NULL);

   private:
       static int Preinit(); 
       void init();
   private:
       // invoked internally by propagate
       int uploadEvent();
       int downloadEvent();
   private:
       // not yet used 
       void indexEvent();
   private:
       int            m_preinit ; 
       OpticksHub*    m_hub ; 
       OpticksIdx*    m_idx ; 
       Opticks*       m_ok ; 
       OpEngine*      m_engine ; 
       OpTracer*      m_tracer ; 
       int            m_placeholder ; 
       
};

#include "OKOP_TAIL.hh"



