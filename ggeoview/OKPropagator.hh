#pragma once

class SLog ; 
template <typename T> class NPY ; 

class OpticksHub ; 
class OpticksIdx ; 
class OpticksViz ; 
class Opticks ; 


#ifdef WITH_OPTIX
class OpEngine ; 
class OKGLTracer ; 
#endif

#include "GGV_API_EXPORT.hh"
#include "GGV_HEAD.hh"

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


**/


class GGV_API OKPropagator {
   public:
       OKPropagator(OpticksHub* hub, OpticksIdx* idx, OpticksViz* viz);
   public:
       void propagate();
       void cleanup();
   public:
       void uploadEvent();
       void downloadEvent();
       void indexEvent();
   private:
       SLog*          m_log ; 
       OpticksHub*    m_hub ; 
       OpticksIdx*    m_idx ; 
       OpticksViz*    m_viz ; 
       Opticks*       m_ok ; 
#ifdef WITH_OPTIX
       OpEngine*      m_engine ; 
       OKGLTracer*    m_tracer ; 
#endif
       int            m_placeholder ; 
       
};

#include "GGV_TAIL.hh"


