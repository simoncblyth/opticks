#pragma once

class SLog ; 
template <typename T> class NPY ; 

class OpticksHub ; 
class Opticks ; 
class OpticksIdx ; 
class Opticks ; 

class OpEngine ; 
class OpTracer ; 

#include "OKOP_API_EXPORT.hh"
#include "OKOP_HEAD.hh"

/**
OpPropagator : compute only okop- equivalent of the interop ok-/OKPropagator 
=============================================================================




**/


class OKOP_API OpPropagator {
   public:
       OpPropagator(OpticksHub* hub, OpticksIdx* idx );
   public:
       void propagate();
       void cleanup();
       void snap();
   public:
       int uploadEvent();
       int downloadEvent();
       void indexEvent();
   private:
       SLog*          m_log ; 
       OpticksHub*    m_hub ; 
       OpticksIdx*    m_idx ; 
       Opticks*       m_ok ; 
       OpEngine*      m_engine ; 
       OpTracer*      m_tracer ; 
       int            m_placeholder ; 
       
};

#include "OKOP_TAIL.hh"



