#pragma once

class SLog ; 
class Opticks ; 
class OpticksHub ; 
class OpticksGen ; 
class OpticksRun ; 
class OpticksIdx; 
class OpEvt ;

class OpPropagator ; 

#include "OKOP_API_EXPORT.hh"
#include "OKOP_HEAD.hh"

/*
OpMgr
======

The highest of high level control, for compute only Opticks 
(no viz, OpenGL).
Used from primary applications such as *OpTest* (okop/tests/OpTest.cc)

*/


class OKOP_API OpMgr {
   public:
       OpMgr(int argc, char** argv, const char* argforced=0 );
       virtual ~OpMgr();
   public:
       void propagate();
       void addGenstep( float* data, unsigned num_float );
       unsigned getNumGensteps() const ;
       unsigned getNumHits() const ;

       void snap();
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
       OpPropagator*  m_propagator ; 
       int            m_count ;  
       OpEvt*         m_opevt ; 
       
};

#include "OKOP_TAIL.hh"

