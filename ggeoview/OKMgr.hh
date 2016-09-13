#pragma once

class SLog ; 
class Opticks ; 
class OpticksHub ; 
class OpticksGen ; 
class OpticksRun ; 
class OpticksIdx; 
class OpticksViz ; 
class OKPropagator ; 

#include "GGV_API_EXPORT.hh"
#include "GGV_HEAD.hh"

class GGV_API OKMgr {
   public:
       OKMgr(int argc, char** argv);
       virtual ~OKMgr();
   public:
       void propagate();
       void visualize();
   private:
       void init();
       void cleanup();
   private:
       SLog*          m_log ; 
       Opticks*       m_ok ; 
       OpticksHub*    m_hub ; 
       OpticksIdx*    m_idx ; 
       int            m_num_event ;  
       OpticksGen*    m_gen ; 
       OpticksRun*    m_run ; 
       OpticksViz*    m_viz ; 
       OKPropagator*  m_propagator ; 
       int            m_count ;  
       
};

#include "GGV_TAIL.hh"

