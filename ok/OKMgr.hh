#pragma once

class SLog ; 
class Opticks ; 
class OpticksHub ; 
class OpticksGen ; 
class OpticksRun ; 
class OpticksIdx; 
class OpticksViz ; 
class OKPropagator ; 

#include "OK_API_EXPORT.hh"
#include "OK_HEAD.hh"

/**
OKMgr
======

Together with OKG4Mgr the highest of high level control.
Used from primary applications such as *OKTest* (ok/tests/OKTest.cc)

**/


class OK_API OKMgr {
   public:
       OKMgr(int argc, char** argv, const char* argforced=0 );
       virtual ~OKMgr();
   public:
       void propagate();
       void visualize();
       int rc() const ; 
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

#include "OK_TAIL.hh"

