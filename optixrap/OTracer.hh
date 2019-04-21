#pragma once

#include "OXPPNS.hh"

class OContext ; 
class Composition ; 
//struct STimes ; 
class BTimes ; 

/**
OTracer
=========

Instance m_otracer is resident of opticksgl/OKGLTracer.hh 


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
       void trace_();
       void report(const char* msg="OTracer::report");
       void setResolutionScale(unsigned int resolution_scale);
       unsigned getResolutionScale() const ;
       unsigned getTraceCount() const ;
       BTimes* getTraceTimes() const ;  
   private:
       void init();

   private:
       OContext*       m_ocontext ; 
       Composition*    m_composition ; 
       optix::Context  m_context ; 
       unsigned        m_resolution_scale ; 

       BTimes*         m_trace_times ; 
       unsigned        m_trace_count ; 
       double          m_trace_prep ; 
       double          m_trace_time ; 

       int             m_entry_index ; 

};




