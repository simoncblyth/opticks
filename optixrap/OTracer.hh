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

#include "OXPPNS.hh"

class Opticks ; 
class OContext ; 
class Composition ; 
class BTimes ; 

/**
OTracer
=========

Instance m_otracer is resident of opticksgl/OKGLTracer.hh 
which coordinates with oglrap/OpticksViz using SRenderer
protocol base.


output_buffer
--------------

OTracer relies on, externally setup 

* output_buffer
* geometry top_object 


* output_buffer typically setup and managed by opticksgl/OFrame.hh
  (done like this as for OpenGL interop the OpenGL buffer needs to be 
  created first, then the OptiX reference to that GPU buffer is made)

* a few tests also "manually" setup an output_buffer 
  eg optixrap/tests/intersect_analytic_test.cc
  but that is not the same format a the pinhole output buffer

oxrap/cu/pinhole_camera.cu
----------------------------

::

   18 rtBuffer<uchar4, 2>   output_buffer;

trace_
--------

* passes composition data (viewpoint, camera param) 
  into OptiX context 

* launches OptiX pinhole camera raytrace writing into 
  pure OptiX output_buffer



**/

#include "OXRAP_API_EXPORT.hh"
class OXRAP_API OTracer {
   public:
       OTracer(OContext* ocontext, Composition* composition);
   public:
       double trace_();
       void report(const char* msg="OTracer::report");
       void setResolutionScale(unsigned int resolution_scale);
       unsigned getResolutionScale() const ;
       unsigned getTraceCount() const ;
       BTimes*  getTraceTimes() const ;  
   private:
       void init();

   private:
       OContext*       m_ocontext ; 
       Opticks*        m_ok ; 
       Composition*    m_composition ; 
       optix::Context  m_context ; 
       unsigned        m_resolution_scale ; 

       BTimes*         m_trace_times ; 
       unsigned        m_trace_count ; 
       double          m_trace_prep ; 
       double          m_trace_time ; 

       int             m_entry_index ; 

};




