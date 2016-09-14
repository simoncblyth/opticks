#pragma once

class SLog ; 
class Opticks ; 
class OpticksHub ; 
class OpticksIdx; 
class OpticksGen ; 
class OpticksRun ; 
class CG4 ; 
class CCollector ; 
class OpticksViz ; 
class OKPropagator ; 

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
       void cleanup();
   private:
       SLog*          m_log ; 
       Opticks*       m_ok ; 
       OpticksHub*    m_hub ; 
       OpticksIdx*    m_idx ; 
       int            m_num_event ; 
       OpticksGen*    m_gen ; 
       OpticksRun*    m_run ; 
       CG4*           m_g4 ; 
       CCollector*    m_collector ; 
       OpticksViz*    m_viz ; 
       OKPropagator*  m_propagator ; 
    
};

#include "OKG4_TAIL.hh"

