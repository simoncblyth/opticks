#pragma once

class SLog ; 
template <typename T> class NPY ; 

class OpticksHub ; 
class OpticksIdx ; 
class OpticksViz ; 
class Opticks ; 

class OpEngine ; 
class OKGLTracer ; 

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

**/


class GGV_API OKPropagator {
   public:
       OKPropagator(OpticksHub* hub, OpticksIdx* idx, OpticksViz* viz);
       virtual ~OKPropagator();
   public:
       void propagate();
       void cleanup();
   private:
       void init();
   private:
       SLog*          m_log ; 
       OpticksHub*    m_hub ; 
       OpticksIdx*    m_idx ; 
       OpticksViz*    m_viz ; 
       Opticks*       m_ok ; 
       OpEngine*      m_engine ; 
       OKGLTracer*    m_tracer ; 
       
};

#include "GGV_TAIL.hh"


