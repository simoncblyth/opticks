#pragma once

class SLog ; 
class Opticks ; 
class OpticksHub ; 
class OpticksGen ; 
class OpticksRun ; 
class OpticksIdx; 
class CCollector ; 
class CG4 ; 
class OpticksViz ; 

#ifdef WITH_OPTIX
class OKPropagator ; 
#endif

#include "OKG4_API_EXPORT.hh"
#include "OKG4_HEAD.hh"

class OKG4_API OKG4Mgr {
   public:
       OKG4Mgr(int argc, char** argv);
       virtual ~OKG4Mgr();
  public:
       void propagate();
       void visualize();
   private:
       void init();
       void indexPropagation();
       void cleanup();
   private:
       SLog*          m_log ; 
       Opticks*       m_ok ; 
       OpticksHub*    m_hub ; 
       OpticksIdx*    m_idx ; 
       OpticksGen*    m_gen ; 
       OpticksRun*    m_run ; 
       CG4*           m_g4 ; 
       CCollector*    m_collector ; 
       OpticksViz*    m_viz ; 
#ifdef WITH_OPTIX
       OKPropagator*  m_propagator ; 
#endif
       int            m_placeholder ; 
    
};

#include "OKG4_TAIL.hh"

