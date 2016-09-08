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

class GGV_API OKPropagator {
   public:
       OKPropagator(OpticksHub* hub, OpticksIdx* idx, OpticksViz* viz);
   public:
       void propagate(NPY<float>* gs);
       void indexPropagation();
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


