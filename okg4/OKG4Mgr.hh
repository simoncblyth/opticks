#pragma once

/**
OKG4Mgr
=========

Highest level manager class for full featured 
Opticks running with Geant4 embedded. 

When "--load" option is not used OKG4Mgr holds a CG4 instance.

**/


class SLog ; 
class Opticks ; 
class OpticksHub ; 
class OpticksIdx; 
class OpticksGen ; 
class OpticksRun ; 
class CG4 ; 
class CGenerator ; 
class OpticksViz ; 
class OKPropagator ; 

#include "OKG4_API_EXPORT.hh"
#include "OKG4_HEAD.hh"

class OKG4_API OKG4Mgr {
   public:
       OKG4Mgr(int argc, char** argv);
       virtual ~OKG4Mgr();
  private:  
       int preinit() const ;  
       void init() const ;  
  public:
       void propagate();
       void visualize();
       int rc() const ;
   private:
       void propagate_();
       void cleanup();
   private:
       SLog*          m_log ; 
       Opticks*       m_ok ; 
       int            m_preinit ; 
       OpticksRun*    m_run ; 
       OpticksHub*    m_hub ; 
       bool           m_load ; 
       OpticksIdx*    m_idx ; 
       int            m_num_event ; 
       OpticksGen*    m_gen ; 
       CG4*           m_g4 ; 
       CGenerator*    m_generator ; 
       OpticksViz*    m_viz ; 
       OKPropagator*  m_propagator ; 
    
};

#include "OKG4_TAIL.hh"

