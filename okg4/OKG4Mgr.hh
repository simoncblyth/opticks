#pragma once

class Opticks ; 
class OpticksHub ; 
class OpticksIdx; 
class CCollector ; 
class CG4 ; 
class OpticksViz ; 

#ifdef WITH_OPTIX
class OpEngine ; 
class OpViz ; 
#endif

#include "OKG4_API_EXPORT.hh"
#include "OKG4_HEAD.hh"

class OKG4_API OKG4Mgr {
   public:
       OKG4Mgr(int argc, char** argv);
   private:
       void init();
   public:
       void propagate();
       void indexPropagation();
       void visualize();
       void cleanup();
   private:
       Opticks*       m_ok ; 
       OpticksHub*    m_hub ; 
       OpticksIdx*    m_idx ; 
       CG4*           m_g4 ; 
       CCollector*    m_collector ; 
       OpticksViz*    m_viz ; 
#ifdef WITH_OPTIX
       OpEngine*      m_ope ; 
       OpViz*         m_opv ; 
#endif
       int            m_placeholder ; 
    
};

#include "OKG4_TAIL.hh"

