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


